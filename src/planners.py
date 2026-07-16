"""Grid-based path-planning utilities."""

from collections import deque
from dataclasses import dataclass
import heapq
import time

import numpy as np


GridPoint = tuple[int, int]


@dataclass
class AStarResult:
    """Output returned by the standard A* planner."""

    path: list[GridPoint]
    path_length: float
    expanded_nodes: int
    planning_time: float
    success: bool

@dataclass
class CostAwareAStarResult:
    """Output returned by the spatial-cost-aware A* planner."""

    path: list[GridPoint]
    geometric_length: float
    objective_cost: float
    expanded_nodes: int
    planning_time: float
    success: bool


def path_exists(
    free_space: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
) -> bool:
    """
    Check path existence using breadth-first search.

    The search uses 8-connected movement and prohibits diagonal
    corner cutting.
    """
    if free_space.ndim != 2:
        raise ValueError(
            "free_space must be a two-dimensional array."
        )

    height, width = free_space.shape

    start_row, start_col = start
    goal_row, goal_col = goal

    start_inside = (
        0 <= start_row < height
        and 0 <= start_col < width
    )

    goal_inside = (
        0 <= goal_row < height
        and 0 <= goal_col < width
    )

    if not start_inside or not goal_inside:
        return False

    if not free_space[start] or not free_space[goal]:
        return False

    movements = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    visited = np.zeros_like(
        free_space,
        dtype=bool,
    )

    visited[start] = True
    queue = deque([start])

    while queue:
        current_row, current_col = queue.popleft()

        if (current_row, current_col) == goal:
            return True

        for delta_row, delta_col in movements:
            next_row = current_row + delta_row
            next_col = current_col + delta_col

            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue

            neighbor = (next_row, next_col)

            if visited[neighbor]:
                continue

            if not free_space[neighbor]:
                continue

            is_diagonal = (
                delta_row != 0
                and delta_col != 0
            )

            if is_diagonal:
                adjacent_vertical = (
                    current_row + delta_row,
                    current_col,
                )

                adjacent_horizontal = (
                    current_row,
                    current_col + delta_col,
                )

                if (
                    not free_space[adjacent_vertical]
                    or not free_space[adjacent_horizontal]
                ):
                    continue

            visited[neighbor] = True
            queue.append(neighbor)

    return False


def octile_distance(
    current: GridPoint,
    goal: GridPoint,
) -> float:
    """Return the octile-distance heuristic."""
    row_difference = abs(
        current[0] - goal[0]
    )

    col_difference = abs(
        current[1] - goal[1]
    )

    diagonal_moves = min(
        row_difference,
        col_difference,
    )

    straight_moves = (
        max(
            row_difference,
            col_difference,
        )
        - diagonal_moves
    )

    return (
        np.sqrt(2.0) * diagonal_moves
        + straight_moves
    )


def reconstruct_path(
    came_from: dict[GridPoint, GridPoint],
    goal: GridPoint,
) -> list[GridPoint]:
    """Reconstruct a start-to-goal path."""
    path = [goal]
    current = goal

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_search(
    free_space: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
) -> AStarResult:
    """
    Perform standard A* search on an 8-connected grid.

    Diagonal corner cutting is prohibited.
    """
    start_time = time.perf_counter()

    if free_space.ndim != 2:
        raise ValueError(
            "free_space must be a two-dimensional array."
        )

    height, width = free_space.shape

    for point in (start, goal):
        row, col = point

        if not (
            0 <= row < height
            and 0 <= col < width
        ):
            return AStarResult(
                path=[],
                path_length=np.inf,
                expanded_nodes=0,
                planning_time=(
                    time.perf_counter()
                    - start_time
                ),
                success=False,
            )

    if not free_space[start] or not free_space[goal]:
        return AStarResult(
            path=[],
            path_length=np.inf,
            expanded_nodes=0,
            planning_time=(
                time.perf_counter()
                - start_time
            ),
            success=False,
        )

    movements = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    open_heap: list[
        tuple[float, float, GridPoint]
    ] = []

    heapq.heappush(
        open_heap,
        (
            octile_distance(start, goal),
            0.0,
            start,
        ),
    )

    came_from: dict[
        GridPoint,
        GridPoint,
    ] = {}

    g_cost: dict[
        GridPoint,
        float,
    ] = {
        start: 0.0,
    }

    closed_set: set[GridPoint] = set()
    expanded_nodes = 0

    while open_heap:
        _, popped_g, current = heapq.heappop(
            open_heap
        )

        if current in closed_set:
            continue

        if popped_g > g_cost.get(
            current,
            np.inf,
        ):
            continue

        closed_set.add(current)
        expanded_nodes += 1

        if current == goal:
            path = reconstruct_path(
                came_from,
                goal,
            )

            return AStarResult(
                path=path,
                path_length=g_cost[goal],
                expanded_nodes=expanded_nodes,
                planning_time=(
                    time.perf_counter()
                    - start_time
                ),
                success=True,
            )

        current_row, current_col = current

        for (
            delta_row,
            delta_col,
            movement_cost,
        ) in movements:
            next_row = current_row + delta_row
            next_col = current_col + delta_col

            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue

            neighbor = (
                next_row,
                next_col,
            )

            if not free_space[neighbor]:
                continue

            is_diagonal = (
                delta_row != 0
                and delta_col != 0
            )

            if is_diagonal:
                adjacent_vertical = (
                    current_row + delta_row,
                    current_col,
                )

                adjacent_horizontal = (
                    current_row,
                    current_col + delta_col,
                )

                if (
                    not free_space[adjacent_vertical]
                    or not free_space[adjacent_horizontal]
                ):
                    continue

            tentative_g = (
                g_cost[current]
                + movement_cost
            )

            if tentative_g >= g_cost.get(
                neighbor,
                np.inf,
            ):
                continue

            came_from[neighbor] = current
            g_cost[neighbor] = tentative_g

            total_cost = (
                tentative_g
                + octile_distance(
                    neighbor,
                    goal,
                )
            )

            heapq.heappush(
                open_heap,
                (
                    total_cost,
                    tentative_g,
                    neighbor,
                ),
            )

    return AStarResult(
        path=[],
        path_length=np.inf,
        expanded_nodes=expanded_nodes,
        planning_time=(
            time.perf_counter()
            - start_time
        ),
        success=False,
    )


def validate_path(
    free_space: np.ndarray,
    path: list[GridPoint],
    start: GridPoint,
    goal: GridPoint,
) -> bool:
    """Verify that a path is connected and collision-free."""
    if not path:
        return False

    if path[0] != start or path[-1] != goal:
        return False

    height, width = free_space.shape

    for index, current in enumerate(path):
        row, col = current

        if not (
            0 <= row < height
            and 0 <= col < width
        ):
            return False

        if not free_space[current]:
            return False

        if index == 0:
            continue

        previous_row, previous_col = (
            path[index - 1]
        )

        row_difference = abs(
            row - previous_row
        )

        col_difference = abs(
            col - previous_col
        )

        if max(
            row_difference,
            col_difference,
        ) != 1:
            return False

        is_diagonal = (
            row_difference == 1
            and col_difference == 1
        )

        if is_diagonal:
            if (
                not free_space[
                    row,
                    previous_col,
                ]
                or not free_space[
                    previous_row,
                    col,
                ]
            ):
                return False

    return True


def calculate_path_length(
    path: list[GridPoint],
) -> float:
    """Calculate the geometric length of a grid path."""
    if len(path) < 2:
        return 0.0

    path_array = np.asarray(
        path,
        dtype=np.float64,
    )

    differences = np.diff(
        path_array,
        axis=0,
    )

    step_lengths = np.linalg.norm(
        differences,
        axis=1,
    )

    return float(
        np.sum(step_lengths)
    )


def cost_aware_astar(
    free_space: np.ndarray,
    traversal_cost: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
) -> CostAwareAStarResult:
    """
    Perform A* search with a spatially varying traversal-cost map.

    The planner minimizes a safety-aware objective instead of only
    geometric path length. Diagonal corner cutting is prohibited.

    Args:
        free_space:
            Boolean grid where True represents traversable cells.
        traversal_cost:
            Positive finite cost for every free cell and infinite cost
            for obstacles or cells outside the road.
        start:
            Start coordinate in (row, column) format.
        goal:
            Goal coordinate in (row, column) format.

    Returns:
        CostAwareAStarResult containing the expert path and statistics.
    """
    start_time = time.perf_counter()

    if free_space.ndim != 2:
        raise ValueError(
            "free_space must be a two-dimensional array."
        )

    if traversal_cost.shape != free_space.shape:
        raise ValueError(
            "traversal_cost and free_space must have the same shape."
        )

    height, width = free_space.shape

    for point in (start, goal):
        row, col = point

        if not (
            0 <= row < height
            and 0 <= col < width
        ):
            return CostAwareAStarResult(
                path=[],
                geometric_length=np.inf,
                objective_cost=np.inf,
                expanded_nodes=0,
                planning_time=time.perf_counter() - start_time,
                success=False,
            )

    if not free_space[start] or not free_space[goal]:
        return CostAwareAStarResult(
            path=[],
            geometric_length=np.inf,
            objective_cost=np.inf,
            expanded_nodes=0,
            planning_time=time.perf_counter() - start_time,
            success=False,
        )

    free_costs = traversal_cost[free_space]

    if (
        free_costs.size == 0
        or not np.all(np.isfinite(free_costs))
        or np.any(free_costs <= 0)
    ):
        raise ValueError(
            "All free-space traversal costs must be finite and positive."
        )

    movements = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    minimum_cell_cost = float(
        np.min(free_costs)
    )

    open_heap: list[
        tuple[float, float, GridPoint]
    ] = []

    initial_heuristic = (
        minimum_cell_cost
        * octile_distance(start, goal)
    )

    heapq.heappush(
        open_heap,
        (
            initial_heuristic,
            0.0,
            start,
        ),
    )

    came_from: dict[
        GridPoint,
        GridPoint,
    ] = {}

    g_cost: dict[
        GridPoint,
        float,
    ] = {
        start: 0.0,
    }

    closed_set: set[GridPoint] = set()
    expanded_nodes = 0

    while open_heap:
        _, popped_g, current = heapq.heappop(
            open_heap
        )

        if current in closed_set:
            continue

        # Ignore outdated heap entries.
        if popped_g > g_cost.get(
            current,
            np.inf,
        ):
            continue

        closed_set.add(current)
        expanded_nodes += 1

        if current == goal:
            path = reconstruct_path(
                came_from,
                goal,
            )

            return CostAwareAStarResult(
                path=path,
                geometric_length=calculate_path_length(path),
                objective_cost=g_cost[goal],
                expanded_nodes=expanded_nodes,
                planning_time=time.perf_counter() - start_time,
                success=True,
            )

        current_row, current_col = current

        for (
            delta_row,
            delta_col,
            movement_distance,
        ) in movements:
            next_row = current_row + delta_row
            next_col = current_col + delta_col

            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue

            neighbor = (
                next_row,
                next_col,
            )

            if not free_space[neighbor]:
                continue

            is_diagonal = (
                delta_row != 0
                and delta_col != 0
            )

            if is_diagonal:
                adjacent_vertical = (
                    current_row + delta_row,
                    current_col,
                )

                adjacent_horizontal = (
                    current_row,
                    current_col + delta_col,
                )

                if (
                    not free_space[adjacent_vertical]
                    or not free_space[adjacent_horizontal]
                ):
                    continue

            # Approximate edge integration using the average cost of
            # the current cell and neighboring cell.
            average_cell_cost = 0.5 * (
                traversal_cost[current]
                + traversal_cost[neighbor]
            )

            step_cost = (
                movement_distance
                * average_cell_cost
            )

            tentative_g = (
                g_cost[current]
                + step_cost
            )

            if tentative_g >= g_cost.get(
                neighbor,
                np.inf,
            ):
                continue

            came_from[neighbor] = current
            g_cost[neighbor] = tentative_g

            heuristic = (
                minimum_cell_cost
                * octile_distance(
                    neighbor,
                    goal,
                )
            )

            total_estimated_cost = (
                tentative_g
                + heuristic
            )

            heapq.heappush(
                open_heap,
                (
                    total_estimated_cost,
                    tentative_g,
                    neighbor,
                ),
            )

    return CostAwareAStarResult(
        path=[],
        geometric_length=np.inf,
        objective_cost=np.inf,
        expanded_nodes=expanded_nodes,
        planning_time=time.perf_counter() - start_time,
        success=False,
    )