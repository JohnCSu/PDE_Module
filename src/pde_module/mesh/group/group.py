import numpy as np
from dataclasses import dataclass, field


@dataclass
class Group:
    """A named group of mesh entities.

    Groups can contain cell, face, edge, and node IDs for labeling
    regions like boundaries or materials.

    Attributes:
        name: Name of the group.
        cell_ids: Array of cell IDs in the group.
        face_ids: Array of face IDs in the group.
        edge_ids: Array of edge IDs in the group.
        node_ids: Array of node IDs in the group.
        int_dtype: NumPy integer dtype for arrays.
        keys: Tuple of valid ID attribute names.
    """

    name: str
    cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    face_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    edge_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    node_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    int_dtype: np.dtype = field(default_factory=lambda: np.int32)
    keys: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize keys and convert dtypes."""
        self.keys = ("cell_ids", "face_ids", "edge_ids", "vertex_ids")
        if self.int_dtype != np.int32:
            for key in self.keys:
                val = getattr(self, key)
                setattr(self, key, np.astype(val, self.int_dtype))

    def __getitem__(self, key: str) -> np.ndarray:
        """Get an ID array by key name."""
        return getattr(self, key)

    @staticmethod
    def union(
        group_1: "Group",
        group_2: "Group",
        name: str | None = None,
        int_dtype: np.dtype = np.int32,
    ) -> "Group":
        """Create a new group that is the union of two groups.

        Args:
            group_1: First group.
            group_2: Second group.
            name: Optional name for the new group.
            int_dtype: NumPy integer dtype.

        Returns:
            New Group containing the union of both groups.
        """
        if name is None:
            name = f"Union_{group_1}_{group_2}"
        return Group(
            name,
            **{key: np.union1d(group_1[key], group_2[key]) for key in group_1.keys},
            int_dtype=int_dtype,
        )

    @staticmethod
    def intersection(
        group_1: "Group",
        group_2: "Group",
        name: str | None = None,
        int_dtype: np.dtype = np.int32,
    ) -> "Group":
        """Create a new group that is the intersection of two groups.

        Args:
            group_1: First group.
            group_2: Second group.
            name: Optional name for the new group.
            int_dtype: NumPy integer dtype.

        Returns:
            New Group containing the intersection of both groups.
        """
        if name is None:
            name = f"Intersection_{group_1}_{group_2}"
        return Group(
            name,
            **{key: np.intersect1d(group_1[key], group_2[key]) for key in group_1.keys},
            int_dtype=int_dtype,
        )
