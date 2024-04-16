import equinox as eqx
import time
from pathlib import Path
from typing import Union
import abc


class AbstractModel(eqx.Module):
    """
        A subclass of eqx.Module, but with save and load to/from file implemented
		Also has wrappers for eqx.partition and eqx.combine, to be overwritten in further
		subclasses to allow freezing of weights during training

    """
    @abc.abstractmethod
    def __call__(self):
        pass

    def partition(self):
        """
        Behaves like eqx.partition. Overwrite in subclasses to account for hard coded array parameters

        Returns
        -------
        diff : PyTree
            PyTree of same structure as AbstractModel, with all non trainable parameters set to None
        static : PyTree
            PyTree of same structure as AbstractModel with all trainable parameters set to None

        """
        diff,static = eqx.partition(self,eqx.is_array)
        return diff,static

    def combine(self,diff,static):
        """
        Wrapper for eqx.combine

        Parameters
        ----------
        diff : PyTree
            PyTree of same structure as AbstractModel, with all non trainable parameters set to None
        static : PyTree
            PyTree of same structure as AbstractModel, with all trainable parameters set to None

        """
        self = eqx.combine(diff,static)




    def save(self, path: Union[str, Path], overwrite: bool = False):
        """
        Wrapper for saving model via eqx.tree_serialise_leaves. Taken from https://github.com/google/jax/issues/2116

        Parameters
        ----------
        path : Union[str, Path]
            path to filename.
        overwrite : bool, optional
            Overwrite existing filename. The default is False.

        Raises
        ------
        RuntimeError
            file already exists.

        Returns
        -------
        None.

        """
        suffix = ".eqx"
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')
        eqx.tree_serialise_leaves(path,self)


    def load(self, path: Union[str, Path]):
        """
        Wrapper for loading model via eqx.tree_deserialise_leaves

        Parameters
        ----------
        path : Union[str, Path]
            path to filename.

        Raises
        ------
        ValueError
            Not a file or incorrect file type.

        Returns
        -------
        AbstractModel
            AbstractModel loaded from pickle.

        """
        suffix = ".eqx"
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != suffix:
            raise ValueError(f'Not a {suffix} file: {path}')
        return eqx.tree_deserialise_leaves(path,self)