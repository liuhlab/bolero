from typing import List, Optional, Tuple

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

from bolero.tl.topic import CistopicLDAModel


def _subset_list(target_list, index_list):
    X = list(map(target_list.__getitem__, index_list))
    return X


def evaluate_models(
    models: List["CistopicLDAModel"],
    select_model: Optional[int] = None,
    return_model: Optional[bool] = True,
    metrics: Optional[str] = [
        "Minmo_2011",
        "loglikelihood",
        "Cao_Juan_2009",
        "Arun_2010",
    ],
    min_topics_coh: Optional[int] = 5,
    plot: Optional[bool] = True,
    figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
    plot_metrics: Optional[bool] = False,
    save: Optional[str] = None,
):
    """
    Model selection based on model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)).

    Parameters
    ----------
    models: list of :class:`CistopicLDAModel`
        A list containing cisTopic LDA models, as returned from run_cgs_models or run_cgs_modelsMallet.
    selected_model: int, optional
        Integer indicating the number of topics of the selected model. If not provided, the best model will be selected automatically based on the model quality metrics. Default: None.
    return_model: bool, optional
        Whether to return the selected model as :class:`CistopicLDAModel`
    metrics: list of str
        Metrics to use for plotting and model selection:
            Minmo_2011: Uses the average model coherence as calculated by Mimno et al (2011). In order to reduce the impact of the number of topics, we calculate the average coherence based on the top selected average values. The better the model, the higher coherence.
            log-likelihood: Uses the log-likelihood in the last iteration as calculated by Griffiths and Steyvers (2004). The better the model, the higher the log-likelihood.
            Arun_2010: Uses a divergence-based metric as in Arun et al (2010) using the topic-region distribution, the cell-topic distribution and the cell coverage. The better the model, the lower the metric.
            Cao_Juan_2009: Uses a density-based metric as in Cao Juan et al (2009) using the topic-region distribution. The better the model, the lower the metric.
        Default: all metrics.
    min_topics_coh: int, optional
        Minimum number of topics on a topic to use its coherence for model selection. Default: 5.
    plot: bool, optional
        Whether to return plot to the console. Default: True.
    figsize: tuple, optional
                Size of the figure. Default: (6.4, 4.8)
    plot_metrics: bool, optional
        Whether to plot metrics independently. Default: False.
    save: str, optional
        Output file to save plot. Default: None.

    Return
    ------
    plot
        Plot with the combined metrics in which the best model should have high values for all metrics (Arun_2010 and Cao_Juan_2011 are inversed).

    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).

    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235

    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.

    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.
    """
    models = [models[i] for i in np.argsort([m.n_topic for m in models])]
    all_topics = sorted([models[x].n_topic for x in range(0, len(models))])
    metrics_dict = {}
    fig = plt.figure(figsize=figsize)
    if "Minmo_2011" in metrics:
        in_index = [
            i for i in range(len(all_topics)) if all_topics[i] >= min_topics_coh
        ]
    if "Arun_2010" in metrics:
        arun_2010 = [
            models[index].metrics.loc["Metric", "Arun_2010"]
            for index in range(0, len(all_topics))
        ]
        arun_2010_negative = [-x for x in arun_2010]
        arun_2010_rescale = (arun_2010_negative - min(arun_2010_negative)) / (
            max(arun_2010_negative) - min(arun_2010_negative)
        )
        if "Minmo_2011" in metrics:
            metrics_dict["Arun_2010"] = np.array(
                _subset_list(arun_2010_rescale, in_index)
            )
        else:
            metrics_dict["Arun_2010"] = arun_2010_rescale
        plt.plot(
            all_topics,
            arun_2010_rescale,
            linestyle="--",
            marker="o",
            label="Inv_Arun_2010",
        )

    if "Cao_Juan_2009" in metrics:
        Cao_Juan_2009 = [
            models[index].metrics.loc["Metric", "Cao_Juan_2009"]
            for index in range(0, len(all_topics))
        ]
        Cao_Juan_2009_negative = [-x for x in Cao_Juan_2009]
        Cao_Juan_2009_rescale = (
            Cao_Juan_2009_negative - min(Cao_Juan_2009_negative)
        ) / (max(Cao_Juan_2009_negative) - min(Cao_Juan_2009_negative))
        if "Minmo_2011" in metrics:
            metrics_dict["Cao_Juan_2009"] = np.array(
                _subset_list(Cao_Juan_2009_rescale, in_index)
            )
        else:
            metrics_dict["Cao_Juan_2009"] = Cao_Juan_2009_rescale
        plt.plot(
            all_topics,
            Cao_Juan_2009_rescale,
            linestyle="--",
            marker="o",
            label="Inv_Cao_Juan_2009",
        )

    if "Minmo_2011" in metrics:
        Mimno_2011 = [
            models[index].metrics.loc["Metric", "Mimno_2011"]
            for index in range(0, len(all_topics))
        ]
        Mimno_2011 = _subset_list(Mimno_2011, in_index)
        Mimno_2011_all_topics = _subset_list(all_topics, in_index)
        Mimno_2011_rescale = (Mimno_2011 - min(Mimno_2011)) / (
            max(Mimno_2011) - min(Mimno_2011)
        )
        metrics_dict["Minmo_2011"] = np.array(Mimno_2011_rescale)
        plt.plot(
            Mimno_2011_all_topics,
            Mimno_2011_rescale,
            linestyle="--",
            marker="o",
            label="Mimno_2011",
        )

    if "loglikelihood" in metrics:
        loglikelihood = [
            models[index].metrics.loc["Metric", "loglikelihood"]
            for index in range(0, len(all_topics))
        ]
        loglikelihood_rescale = (loglikelihood - min(loglikelihood)) / (
            max(loglikelihood) - min(loglikelihood)
        )
        if "Minmo_2011" in metrics:
            metrics_dict["loglikelihood"] = np.array(
                _subset_list(loglikelihood_rescale, in_index)
            )
        else:
            metrics_dict["loglikelihood"] = loglikelihood_rescale
        plt.plot(
            all_topics,
            loglikelihood_rescale,
            linestyle="--",
            marker="o",
            label="Loglikelihood",
        )

    if select_model is None:
        combined_metric = sum(metrics_dict.values())
        if "Minmo_2011" in metrics:
            best_model = Mimno_2011_all_topics[
                combined_metric.tolist().index(max(combined_metric))
            ]
        else:
            best_model = all_topics[
                combined_metric.tolist().index(max(combined_metric))
            ]
    else:
        combined_metric = None
        best_model = select_model

    plt.axvline(best_model, linestyle="--", color="grey")
    plt.xlabel("Number of topics\nOptimal number of topics: " + str(best_model))
    plt.ylabel("Rescaled metric")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        pdf.savefig(fig, bbox_inches="tight")
    if plot is True:
        plt.show()
    else:
        plt.close(fig)

    if plot_metrics:
        if "Arun_2010" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, arun_2010, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Arun_2010 - Minimize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

        if "Cao_Juan_2009" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, Cao_Juan_2009, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Cao_Juan_2009 - Minimize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()
        if "Minmo_2011" in metrics:
            fig = plt.figure()
            plt.plot(Mimno_2011_all_topics, Mimno_2011, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Mimno_2011 - Maximize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

        if "loglikelihood" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, loglikelihood, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Loglikelihood - Maximize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

    if save is not None:
        pdf.close()

    if return_model:
        return models[all_topics.index(best_model)]
