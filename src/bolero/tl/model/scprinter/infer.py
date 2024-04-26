import torch

from bolero.tl.footprint.footprint import postprocess_footprint


class BatchInference:
    """
    Perform batch inference using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model used for inference.
    postprocess : bool, optional
        Flag indicating whether to apply post-processing to the output. Default is True.

    Returns
    -------
    dict
        A dictionary containing the input data along with the inferred results.
    """

    def __init__(self, model: torch.nn.Module, postprocess: bool = True):
        self.model = model
        self.postprocess = postprocess

    def __call__(self, data: dict) -> dict:
        """
        Perform batch inference on the given data.

        Parameters
        ----------
        data : dict
            A dictionary containing the input data.

        Returns
        -------
        dict
            A dictionary containing the input data along with the inferred results.
        """
        one_hot = data["dna_one_hot"]
        with torch.inference_mode():
            footprint, coverage = self.model(one_hot)
        if self.postprocess:
            footprint = postprocess_footprint(footprint=footprint, smooth_radius=5)
        data["footprint"] = footprint
        data["coverage"] = coverage.numpy()
        return data
