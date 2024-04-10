import logging
import math
import pathlib
import shlex
import subprocess
import tempfile
from itertools import chain
from typing import List, Optional, Union

import anndata
import joblib
import numpy as np
import pandas as pd
import ray
import tmtoolkit
from gensim import corpora, matutils, utils
from gensim.models import basemodel
from gensim.utils import check_output, revdict
from scipy import sparse

try:
    MALLET_PATH = subprocess.check_output(["which", "mallet"], encoding="utf8").strip()
    MALLET_PATH = pathlib.Path(MALLET_PATH)
    _mallet_grandparent = MALLET_PATH.parent.parent
    MALLET_JAVA_BASE = (
        "java "
        "-Xmx{mem_gb}g "
        "-ea "
        "-Djava.awt.headless=true "
        "-Dfile.encoding=UTF-8 "
        "-server "
        f"-classpath {_mallet_grandparent}/class:{_mallet_grandparent}/lib/mallet-deps.jar: "
        "{mallet_cmd} "
    )
except subprocess.CalledProcessError:
    MALLET_PATH = "MALLET_PATH_NOT_FOUND"
    print(
        "LDA package Mallet not found. Please install it with conda/mamba install -c conda-forge mallet"
    )

MALLET_COMMAND_MAP = {
    "import-dir": "cc.mallet.classify.tui.Text2Vectors",
    "import-file": "cc.mallet.classify.tui.Csv2Vectors",
    "import-svmlight": "cc.mallet.classify.tui.SvmLight2Vectors",
    "info": "cc.mallet.classify.tui.Vectors2Info",
    "train-classifier": "cc.mallet.classify.tui.Vectors2Classify",
    "classify-dir": "cc.mallet.classify.tui.Text2Classify",
    "classify-file": "cc.mallet.classify.tui.Csv2Classify",
    "classify-svmlight": "cc.mallet.classify.tui.SvmLight2Classify",
    "train-topics": "cc.mallet.topics.tui.TopicTrainer",
    "infer-topics": "cc.mallet.topics.tui.InferTopics",
    "evaluate-topics": "cc.mallet.topics.tui.EvaluateTopics",
    "prune": "cc.mallet.classify.tui.Vectors2Vectors",
    "split": "cc.mallet.classify.tui.Vectors2Vectors",
    "bulk-load": "cc.mallet.util.BulkLoader",
}


def _prepare_binary_matrix(data):
    # binary_matrix is a matrix containing cells as columns and regions as rows, following the cisTopic format
    if isinstance(data, anndata.AnnData):
        binary_matrix = data.X.T
        cell_names = data.obs_names
        region_names = data.var_names
    elif isinstance(data, pd.DataFrame):
        binary_matrix = data.values.T
        cell_names = data.index
        region_names = data.columns
    elif isinstance(data, (pathlib.Path, str)):
        assert pathlib.Path(data).exists(), f"{data} does not exist"
        data = anndata.read_h5ad(data)
        binary_matrix = data.X.T
        cell_names = data.obs_names
        region_names = data.var_names
    else:
        raise ValueError(
            "data has to be an anndata.AnnData or a pd.DataFrame or a pathlib.Path or a str path to an h5ad file."
        )

    if isinstance(binary_matrix, np.ndarray):
        binary_matrix = sparse.csc_matrix(binary_matrix)
    elif sparse.issparse(binary_matrix):
        binary_matrix = binary_matrix.tocsc()
    else:
        raise ValueError(
            "binary_matrix has to be a numpy.ndarray or a sparse.csr_matrix"
        )
    return binary_matrix, cell_names, region_names


def loglikelihood(nzw, ndz, alpha, eta):
    """
    Calculate the log-likelihood of the Latent Dirichlet Allocation (LDA) model.

    Parameters
    ----------
    - nzw (numpy.ndarray): The word-topic matrix of shape (n_topics, vocab_size).
    - ndz (numpy.ndarray): The document-topic matrix of shape (D, n_topics).
    - alpha (float): The hyperparameter for the document-topic distribution.
    - eta (float): The hyperparameter for the word-topic distribution.

    Returns
    -------
    - ll (float): The log-likelihood of the LDA model.

    """
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]

    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha * n_topics)) * D
    const_ll = (
        vocab_size * math.lgamma(eta) - math.lgamma(eta * vocab_size)
    ) * n_topics

    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta * vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll = math.lgamma(nzw[k, w] + eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)

    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha * n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll = math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll = doc_ll - const_prior + topic_ll - const_ll
    return ll


def convert_input(
    data, output_prefix, train_mallet_file=None, train_id2word_file=None, mem_gb=4
):
    """
    Convert sparse.csc_matrix to Mallet format and save it to a binary file, also save the id2word dictionary.

    Parameters
    ----------
    data : sparse.csc_matrix
        binary matrix containing cells/documents as columns and regions/words as rows.
    output_prefix : str
        Prefix to save the output files.
    train_mallet_file : str, optional
        Path to the corpus in Mallet format. Provided if the model is being trained and the corpus is being used for inference. Default: None.
    train_id2word_file : str, optional
        Path to the id2word dictionary. Provided if the model is being trained and the corpus is being used for inference. Default: None.
    mem_gb : int, optional
        Memory to use in GB. Default: 4.

    Returns
    -------
    mallet_path : str
        Path to the corpus in Mallet format.
    id2word_path : str
        Path to the id2word dictionary.
    """
    corpus = matutils.Sparse2Corpus(data)

    if train_mallet_file is not None or train_id2word_file is not None:
        trained = True
        assert (
            train_id2word_file is not None
        ), "train_id2word_file and train_mallet_file have to be provided together"
        assert (
            train_mallet_file is not None
        ), "train_id2word_file and train_mallet_file have to be provided together"
        train_mallet_path = pathlib.Path(train_mallet_file)
        # this will change the permissions of the file to read-only, because the mallet java command will try
        # to write to this file everytime it is used to keep the mallet dictionary up-to-date
        # however, this create a problem when running the mallet command in parallel
        # each parallel process will try to write to the same file and the mallet command will fail
        # also in our usecase the mallet file does not need to be updated if the model is already trained
        # also, chmod will not work in gcsfuse mounted directories, the current solution is to copy the file to a temp file
        train_mallet_path.chmod(0o444)

        id2word_path = pathlib.Path(train_id2word_file)
        id2word = joblib.load(id2word_path)
    else:
        trained = False
        names_dict = {x: str(x) for x in range(data.shape[0])}
        id2word = corpora.Dictionary.from_corpus(corpus, names_dict)
        id2word_path = pathlib.Path(f"{output_prefix}_corpus.id2word")
        temp_id2word_path = pathlib.Path(f"{output_prefix}_corpus.id2word.temp")

    txt_path = pathlib.Path(f"{output_prefix}_corpus.txt")
    mallet_path = pathlib.Path(f"{output_prefix}_corpus.mallet")
    temp_mallet_path = pathlib.Path(f"{output_prefix}_corpus.mallet.temp")

    if trained:
        if mallet_path.exists():
            return mallet_path, id2word_path
    else:
        if mallet_path.exists() and id2word_path.exists():
            return mallet_path, id2word_path

    with utils.open(txt_path, "wb") as fout:
        for docno, doc in enumerate(corpus):
            tokens = chain.from_iterable(
                [id2word[tokenid]] * int(cnt) for tokenid, cnt in doc
            )
            fout.write(utils.to_utf8("{} 0 {}\n".format(docno, " ".join(tokens))))

    # save mallet binary file
    _mallet_cmd = "import-file"
    _mallet_cmd_base = MALLET_JAVA_BASE.format(
        mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd]
    )
    cmd = (
        f"{_mallet_cmd_base} "
        "--preserve-case "
        "--keep-sequence "
        '--remove-stopwords --token-regex "\\S+" '
        f"--input {txt_path} "
        f"--output {temp_mallet_path} "
    )
    if trained:
        cmd += f"--use-pipe-from {train_mallet_path} "
    try:
        check_output(args=cmd, shell=True, stderr=subprocess.STDOUT, encoding="utf8")
    except subprocess.CalledProcessError as e:
        # Here Java will raise an error about the train-mallet-file being read-only and permission denied
        # This is expected because we modified the permissions of the file to read-only above
        # This is the last step of preparing the input file, the input mallet file is already created
        # so we can ignore this error
        # This is a ugly way to handle this error, but it is the only way to avoid the error (chatgpt said that)
        if "java.io.FileNotFoundException" in e.output:
            # print('there there, it is ok')
            # The java line that raises this erore is:
            # https://github.com/mimno/Mallet/blob/master/src/cc/mallet/classify/tui/Csv2Vectors.java#L336
            pass
        else:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            ) from e
    txt_path.unlink()

    if not trained:
        # dump id2word to a file
        joblib.dump(id2word, temp_id2word_path)
        temp_id2word_path.rename(id2word_path)

    temp_mallet_path.rename(mallet_path)
    return mallet_path, id2word_path


class CistopicLDAModel:
    """
    cisTopic LDA model class

    :class:`cistopicLdaModel` contains model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)), topic quality metrics (coherence, marginal distribution and total number of assignments), cell-topic and topic-region distribution, model parameters and model dimensions.

    Parameters
    ----------
    metrics: pd.DataFrame
        :class:`pd.DataFrame` containing model quality metrics, including model coherence (adaptation from Mimno et al., 2011), log-likelihood and density and divergence-based methods (Cao Juan et al., 2009; Arun et al., 2010).
    coherence: pd.DataFrame
        :class:`pd.DataFrame` containing the coherence of each topic (Mimno et al., 2011).
    marginal_distribution: pd.DataFrame
        :class:`pd.DataFrame` containing the marginal distribution for each topic. It can be interpreted as the importance of each topic for the whole corpus.
    topic_ass: pd.DataFrame
        :class:`pd.DataFrame` containing the total number of assignments per topic.
    cell_topic: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with cells as columns, topics as rows and the probability of each topic in each cell as values.
    topic_region: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with topics as columns, regions as rows and the probability of each region in each topic as values.
    parameters: pd.DataFrame
        :class:`pd.DataFrame` containing parameters used for the model.
    n_cells: int
        Number of cells in the model.https://www.google.com/maps/place/The+Home+Depot/data=!4m7!3m6!1s0x89e378259073c585:0xf1c58d25004b2b2d!8m2!3d42.3620232!4d-71.1560237!16s%2Fg%2F1tp1ztzr!19sChIJhcVzkCV444kRLStLACWNxfE?authuser=0&hl=en&rclk=1
    n_regions: int
        Number of regions in the model.
    n_topic: int
        Number of topics in the model.

    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).

    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.

    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.

    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.
    """

    def __init__(
        self,
        metrics: pd.DataFrame,
        coherence: pd.DataFrame,
        marg_topic: pd.DataFrame,
        topic_ass: pd.DataFrame,
        cell_topic: pd.DataFrame,
        topic_region: pd.DataFrame,
        parameters: pd.DataFrame,
    ):
        self.metrics = metrics
        self.coherence = coherence
        self.marg_topic = marg_topic
        self.topic_ass = topic_ass
        self.cell_topic = cell_topic
        self.cell_topic_harmony = []
        self.topic_region = topic_region
        self.parameters = parameters
        self.n_cells = cell_topic.shape[1]
        self.n_regions = topic_region.shape[0]
        self.n_topic = cell_topic.shape[0]

    def __str__(self):
        descr = f"CistopicLDAModel with {self.n_topic} topics and n_cells × n_regions = {self.n_cells} × {self.n_regions}"
        return descr


class LDAMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    """Run LDA with Mallet"""

    def __init__(
        self,
        corpus_mallet_path: str,
        id2word_path: str,
    ):
        """
        Wrapper class to run LDA models with Mallet.

        This class has been adapted from gensim
        (https://github.com/RaRe-Technologies/gensim/blob/27bbb7015dc6bbe02e00bb1853e7952ac13e7fe0/gensim/models/wrappers/ldamallet.py).

        Parameters
        ----------
        corpus_mallet_path : str
            Path to the corpus in Mallet format.
        id2word_path : str
            Path to the id2word dictionary.
        """
        self.corpus_mallet_path = corpus_mallet_path
        self.id2word_path = id2word_path
        self.id2word = joblib.load(id2word_path)

        self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("Cannot compute LDA over an empty collection (no terms)")

        self.word_topics = None
        self.output_prefix = None
        return

    def train(
        self,
        output_prefix,
        num_topics,
        alpha=50,
        eta=0.1,
        optimize_interval=0,
        topic_threshold=0.0,
        iterations=150,
        random_seed=555,
        n_cpu=8,
        mem_gb=16,
    ):
        """
        Train Mallet LDA.

        Parameters
        ----------
        output_prefix : str
            Prefix to save the output files.
        num_topics : int
            The number of topics to use in the model.
        alpha : float, optional
            alpha value for mallet train-topics. Default: 50.
        eta : float, optional
            beta value for mallet train-topics. Default: 0.1.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization). Default: 0.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic. Default: 0.0.
        iterations : int, optional
            Number of training iterations. Default: 150.
        random_seed : int, optional
            Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
        n_cpu : int, optional
            Number of threads that will be used for training. Default: 1.
        mem_gb : int, optional
            Memory to use in GB. Default: 16.
        """
        state_path = f"{output_prefix}_state.mallet.gz"
        doctopics_path = f"{output_prefix}_doctopics.txt"
        inferencer_path = f"{output_prefix}_inferencer.mallet"
        topickeys_path = f"{output_prefix}_topickeys.txt"

        _mallet_cmd = "train-topics"
        _mallet_cmd_base = MALLET_JAVA_BASE.format(
            mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd]
        )
        cmd = (
            f"{_mallet_cmd_base} "
            f"--input {self.corpus_mallet_path} "
            f"--num-topics {num_topics} "
            f"--alpha {alpha} "
            f"--beta {eta} "
            f"--optimize-interval {optimize_interval} "
            f"--num-threads {int(n_cpu*2)} "
            f"--num-iterations {iterations} "
            f"--output-state {state_path} "
            f"--output-doc-topics {doctopics_path} "
            f"--output-topic-keys {topickeys_path} "
            f"--inferencer-filename {inferencer_path} "
            f"--doc-topics-threshold {topic_threshold} "
            f"--random-seed {random_seed}"
        )
        try:
            subprocess.check_output(
                args=shlex.split(cmd), shell=False, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            ) from e
        self.word_topics = self.load_word_topics(
            num_topics=num_topics, state_path=state_path
        )
        self.output_prefix = output_prefix
        return

    @classmethod
    def parallel_infer(
        cls,
        data,
        model_dirs,
        topic_threshold=0.0,
        num_iterations=300,
        random_seed=555,
        mem_gb=16,
    ):
        """
        Infer topics for new data in parallel.

        Parameters
        ----------
        data : sparse.csr_matrix
            Binary matrix containing cell/document as columns and regions/words as rows.
        model_dirs : list of str
            List of paths to the model directories.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic. Default: 0.0.
        num_iterations : int, optional
            Number of training iterations. Default: 300.
        random_seed : int, optional
            Random seed to ensure consistent results. Default: 555.
        mem_gb : int, optional
            Memory to use in GB. Default: 16.
        """
        if isinstance(model_dirs, (str, pathlib.Path)):
            model_dirs = [model_dirs]
        model_dirs = [pathlib.Path(x).absolute() for x in model_dirs]

        # make sure all model_dir names are unique, because we will use them to name the keys in the model_dict
        model_names = [x.name for x in model_dirs]
        assert len(model_names) == len(
            set(model_names)
        ), "model_dirs names are not unique, please rename them to unique names"

        with tempfile.TemporaryDirectory(prefix="bolero_") as parent_temp_dir:
            model_dict = {}
            for model_dir in model_dirs:
                model_temp_dir = tempfile.mkdtemp(
                    dir=parent_temp_dir, prefix=model_dir.name
                )
                model_files = {
                    "inferencer": {},
                    "train_mallet": pathlib.Path(
                        f"{model_temp_dir}/train_corpus.mallet"
                    ),
                    "train_id2word": model_dir / "train_corpus.id2word",
                }
                actual_train_mallet = model_dir / "train_corpus.mallet"
                assert (
                    actual_train_mallet.exists()
                ), f"train_corpus.mallet file does not exist in {model_dir}"
                # copy the mallet file to temp path and make it read only
                subprocess.run(
                    ["cp", actual_train_mallet, model_files["train_mallet"]], check=True
                )
                model_files["train_mallet"].chmod(0o444)

                assert model_files[
                    "train_id2word"
                ].exists(), f"train_corpus.id2word file does not exist in {model_dir}"

                inferencer_paths = list(
                    pathlib.Path(model_dir).rglob("model*/*inferencer.mallet")
                )
                for inferencer_path in inferencer_paths:
                    topic_model_name = inferencer_path.parent.name
                    model_dir_name = model_dir.name
                    # inferencer name will be unique
                    model_files["inferencer"][
                        f"{model_dir_name}_{topic_model_name}"
                    ] = str(inferencer_path)
                model_dict[model_temp_dir] = model_files

            data, cell_names, _ = _prepare_binary_matrix(data)
            data_remote = ray.put(data)

            @ray.remote(num_cpus=2)
            def _remote_convert_input(
                data,
                chunk_start,
                chunk_end,
                temp_dir,
                train_mallet_file,
                train_id2word_file,
                mem_gb=16,
            ):
                # get a random dir to save the mallet files
                temp_prefix = f"{temp_dir}/infer_{chunk_start}_{chunk_end}"
                _data = data[:, chunk_start:chunk_end]
                mallet_path, _ = convert_input(
                    data=_data,
                    output_prefix=temp_prefix,
                    mem_gb=mem_gb,
                    train_mallet_file=train_mallet_file,
                    train_id2word_file=train_id2word_file,
                )
                return mallet_path

            @ray.remote(num_cpus=1)
            def _remote_infer(mallet_path, inferencer_path, temp_prefix):
                LDAMallet.infer(
                    mallet_path=mallet_path,
                    inferencer_path=inferencer_path,
                    output_path=f"{temp_prefix}_doctopics.txt",
                    topic_threshold=topic_threshold,
                    num_iterations=num_iterations,
                    random_seed=random_seed,
                    mem_gb=mem_gb,
                )
                # load doc_topic table
                doctopics_path = f"{temp_prefix}_doctopics.txt"
                doc_topic = (
                    pd.read_csv(doctopics_path, header=None, sep="\t", comment="#")
                    .iloc[:, 2:]
                    .to_numpy()
                )
                return doc_topic

            # get the number of cpu available and adjust the chunk size
            n_cpu = int(ray.available_resources()["CPU"])
            chunk_size = max(100, (data.shape[1] + n_cpu) // int(n_cpu / 2))

            # convert input for each model, this is required as the train_mallet and train_id2word files are different for each model
            futures = {}
            for model_temp_dir, model_files in model_dict.items():
                # split the data in chunks and prepare inputs
                _futures = [
                    _remote_convert_input.remote(
                        data=data_remote,
                        temp_dir=model_temp_dir,
                        chunk_start=chunk_start,
                        chunk_end=min(chunk_start + chunk_size, data.shape[1]),
                        train_mallet_file=model_files["train_mallet"],
                        train_id2word_file=model_files["train_id2word"],
                        mem_gb=mem_gb,
                    )
                    for chunk_start in range(0, data.shape[1], chunk_size)
                ]
                futures[model_temp_dir] = _futures
            # the mallet paths for each model
            mallet_paths_dict = {}
            for model_temp_dir, _futures in futures.items():
                mallet_paths = ray.get(_futures)
                mallet_paths_dict[model_temp_dir] = mallet_paths

            # run the inference in parallel for each inferencer on each chunk
            inferencer_future_dict = {}
            for model_temp_dir, model_files in model_dict.items():
                mallet_paths = mallet_paths_dict[model_temp_dir]
                for inferencer_name, inferencer_path in model_files[
                    "inferencer"
                ].items():
                    temp_dir = tempfile.mkdtemp(
                        dir=parent_temp_dir, prefix=inferencer_name
                    )
                    futures = [
                        _remote_infer.remote(
                            mallet_path=mallet_path,
                            inferencer_path=inferencer_path,
                            temp_prefix=f"{temp_dir}/{pathlib.Path(mallet_path).stem}",
                        )
                        for mallet_path in mallet_paths
                    ]
                    inferencer_future_dict[inferencer_name] = futures

            # get the results
            doc_topic_dict = {}
            for name, futures in inferencer_future_dict.items():
                doc_topic = np.concatenate(ray.get(futures), axis=0)
                doc_topic = pd.DataFrame(
                    doc_topic,
                    index=cell_names,
                    columns=[f"topic{i}" for i in range(doc_topic.shape[1])],
                )
                doc_topic_dict[name] = doc_topic
        return doc_topic_dict

    @classmethod
    def infer(
        cls,
        mallet_path,
        inferencer_path,
        output_path,
        topic_threshold=0.0,
        num_iterations=300,
        random_seed=555,
        mem_gb=16,
    ):
        """
        Infer topics for new documents.

        Parameters
        ----------
        mallet_path: str
            Path to the corpus in Mallet format.
        inferencer_path: str
            Path to the inferencer file.
        output_prefix: str
            Prefix to save the output files.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic. Default: 0.0.
        num_iterations : int, optional
            Number of training iterations. Default: 300.
        random_seed: int, optional
            Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
        mem_gb: int, optional
            Memory to use in GB. Default: 16.
        """
        _mallet_cmd = "infer-topics"
        _mallet_cmd_base = MALLET_JAVA_BASE.format(
            mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd]
        )
        cmd = (
            f"{_mallet_cmd_base} "
            f"--inferencer {inferencer_path} "
            f"--input {mallet_path} "
            f"--output-doc-topics {output_path} "
            f"--doc-topics-threshold {topic_threshold} "
            f"--num-iterations {num_iterations} "
            f"--random-seed {random_seed} "
        )
        try:
            subprocess.check_output(
                args=shlex.split(cmd), shell=False, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            ) from e
        return

    def load_word_topics(self, num_topics, state_path):
        """
        Load words X topics matrix from :meth:`gensim.models.wrappers.LDAMallet.LDAMallet.fstate` file.

        Returns
        -------
        np.ndarray
            Matrix words X topics.
        """
        logger = logging.getLogger("LDAMalletWrapper")
        logger.info("loading assigned topics from %s", state_path)
        word_topics = np.zeros((num_topics, self.num_terms), dtype=np.float64)
        if hasattr(self.id2word, "token2id"):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)

        with utils.open(state_path, "rb") as fin:
            _ = next(fin)  # header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert (
                len(self.alpha) == num_topics
            ), "Mismatch between MALLET vs. requested topics"
            _ = next(fin)  # beta
            for _, line in enumerate(fin):
                line = utils.to_unicode(line)
                *_, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    def get_topics(self):
        """
        Get topics X words matrix.

        Returns
        -------
        np.ndarray
            Topics X words matrix, shape `num_topics` x `vocabulary_size`.
        """
        doctopics_path = f"{self.output_prefix}_doctopics.txt"
        doc_topic = (
            pd.read_csv(doctopics_path, header=None, sep="\t").iloc[:, 2:].to_numpy()
        )
        topic_word = self.word_topics
        norm_topic_word = topic_word / topic_word.sum(axis=1)[:, None]
        return doc_topic, norm_topic_word


def train_lda(
    data: Union[anndata.AnnData, pd.DataFrame, pathlib.Path, str],
    output_dir,
    n_topics: List[int],
    n_iter: Optional[int] = 500,
    random_state: Optional[int] = 555,
    alpha: Optional[float] = 50,
    eta: Optional[float] = 0.1,
    top_topics_coh: Optional[int] = 5,
    mallet_cpu=8,
    mem_gb=16,
):
    """
    Run Latent Dirichlet Allocation per model as implemented in Mallet (McCallum, 2002).

    Parameters
    ----------
    data: Union[anndata.AnnData, pd.DataFrame, pathlib.Path, str]
        Data matrix containing cells as rows and regions as columns.
        If an anndata.AnnData is provided, the data matrix will be extracted from data.X.T.
        If a pd.DataFrame is provided, the data matrix will be extracted from data.values.T.
        If a pathlib.Path or a str is provided, the data matrix will be loaded as anndata.AnnData
        and the data matrix will be extracted from data.X.T.
    n_topics: list of int
        A list containing the number of topics to use in each model.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model,
        the coherence will be calculated as the average of the top coherence values. Default: 5.
    mallet_cpu: int, optional
        Number of threads for each malles train-topics call. Default: 4.

    Return
    ------
    list of :class:`CistopicLDAModel`
        A list with cisTopic LDA models.

    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_matrix, cell_names, region_names = _prepare_binary_matrix(data)

    # make sure the binary matrix is binary
    binary_matrix.data = np.ones_like(binary_matrix.data, dtype="int8")

    # create mallet file and id2word dictionary
    corpus_mallet_path, id2word_path = convert_input(
        data=binary_matrix, output_prefix=output_dir / "train", mem_gb=mem_gb
    )

    @ray.remote(num_cpus=mallet_cpu)
    def _remote_run_cgs_model_mallet(*args, **kwargs):
        return run_cgs_model_mallet(*args, **kwargs)

    # save the corpus to disk and call mallet in parallel
    binary_matrix_remote = ray.put(binary_matrix)
    futures = [
        _remote_run_cgs_model_mallet.remote(
            binary_matrix=binary_matrix_remote,
            corpus_mallet_path=corpus_mallet_path,
            id2word_path=id2word_path,
            output_dir=output_dir / f"model{n_topic}",
            n_topics=n_topic,
            cell_names=cell_names,
            region_names=region_names,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha,
            eta=eta,
            top_topics_coh=top_topics_coh,
            cpu=mallet_cpu,
            mem_gb=mem_gb,
        )
        for n_topic in n_topics
    ]
    model_list = ray.get(futures)
    return model_list


def run_cgs_model_mallet(
    binary_matrix: sparse.csr_matrix,
    corpus_mallet_path: str,
    id2word_path: str,
    output_dir: str,
    n_topics: List[int],
    cell_names: List[str],
    region_names: List[str],
    n_iter: Optional[int] = 500,
    random_state: Optional[int] = 555,
    alpha: Optional[float] = 50,
    eta: Optional[float] = 0.1,
    top_topics_coh: Optional[int] = 5,
    cpu=8,
    mem_gb=16,
):
    """
    Run Latent Dirichlet Allocation in a model as implemented in Mallet (McCallum, 2002).

    Parameters
    ----------
    binary_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accesible on a cell (otherwise, 0).
    corpus_mallet_path: str
        Path to the corpus in Mallet format.
    id2word_path: str
        Path to the id2word dictionary.
    output_dir: str
        Path to save the model.
    n_topics: list of int
        A list containing the number of topics to use in each model.
    cell_names: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_names: list of str
        List containing region names as ordered in the binary matrix rows.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    cpu: int, optional
        Number of threads that will be used for training. Default: 4.

    Return
    ------
    CistopicLDAModel
        A cisTopic LDA model.

    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = output_dir / "model.lib"
    if final_model_path.exists():
        return joblib.load(final_model_path)

    # Running model
    model = LDAMallet(
        corpus_mallet_path=corpus_mallet_path,
        id2word_path=id2word_path,
    )
    model.train(
        output_prefix=output_dir / "train",
        num_topics=n_topics,
        alpha=alpha,
        eta=eta,
        optimize_interval=0,
        topic_threshold=0.0,
        iterations=n_iter,
        random_seed=random_state,
        n_cpu=cpu,
        mem_gb=mem_gb,
    )

    # Get distributions
    doc_topic, topic_word = model.get_topics()

    # Model evaluation
    cell_cov = np.asarray(binary_matrix.sum(axis=0)).astype(float)
    arun_2010 = tmtoolkit.topicmod.evaluate.metric_arun_2010(
        topic_word, doc_topic, cell_cov
    )
    cao_juan_2009 = tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(topic_word)
    mimno_2011 = tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word,
        dtm=binary_matrix.transpose(),
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )
    topic_word_assig = model.word_topics
    doc_topic_assig = (doc_topic.T * (cell_cov)).T
    ll = loglikelihood(topic_word_assig, doc_topic_assig, alpha, eta)

    # Organize data
    if len(mimno_2011) <= top_topics_coh:
        metrics = pd.DataFrame(
            [arun_2010, cao_juan_2009, np.mean(mimno_2011), ll],
            index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
            columns=["Metric"],
        ).transpose()
    else:
        metrics = pd.DataFrame(
            [
                arun_2010,
                cao_juan_2009,
                np.mean(
                    mimno_2011[
                        np.argpartition(mimno_2011, -top_topics_coh)[-top_topics_coh:]
                    ]
                ),
                ll,
            ],
            index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
            columns=["Metric"],
        ).transpose()
    coherence = pd.DataFrame(
        [range(1, n_topics + 1), mimno_2011], index=["Topic", "Mimno_2011"]
    ).transpose()
    marg_topic = pd.DataFrame(
        [
            range(1, n_topics + 1),
            tmtoolkit.topicmod.model_stats.marginal_topic_distrib(doc_topic, cell_cov),
        ],
        index=["Topic", "Marg_Topic"],
    ).transpose()
    topic_ass = pd.DataFrame.from_records(
        [
            range(1, n_topics + 1),
            list(chain.from_iterable(model.word_topics.sum(axis=1)[:, None])),
        ],
        index=["Topic", "Assignments"],
    ).transpose()
    cell_topic = pd.DataFrame.from_records(
        doc_topic,
        index=cell_names,
        columns=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    topic_region = pd.DataFrame.from_records(
        topic_word,
        columns=region_names,
        index=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    parameters = pd.DataFrame(
        [
            "Mallet",
            n_topics,
            n_iter,
            random_state,
            alpha,
            eta,
            top_topics_coh,
        ],
        index=[
            "package",
            "n_topics",
            "n_iter",
            "random_state",
            "alpha",
            "eta",
            "top_topics_coh",
        ],
        columns=["Parameter"],
    )
    # Create object
    model = CistopicLDAModel(
        metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
    )
    # save model
    joblib.dump(model, final_model_path)
    return model
