"""Interactive Plotly visualizations for 2D benchmark landscapes and optimizer runs."""

from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from metaheuristics.result import OptimizationResult

_BEST_COLOR = "#e45756"
_POPULATION_COLOR = "#a05757"


def evaluate_grid(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``func`` over a 2D grid spanning ``bounds``.

    Returns the ``(X1, X2, Y)`` meshgrid arrays, ready for surface/contour plots.
    """
    (x1_lo, x1_hi), (x2_lo, x2_hi) = bounds
    x1 = np.linspace(x1_lo, x1_hi, resolution)
    x2 = np.linspace(x2_lo, x2_hi, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    Y = func(np.array([X1, X2]))
    return X1, X2, Y


def plot_surface(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]] | None = None,
    resolution: int = 200,
    title: str | None = None,
) -> go.Figure:
    """Interactive 3D surface of a benchmark landscape (rotate/zoom in the notebook)."""
    bounds = bounds if bounds is not None else getattr(func, "bounds", None)
    if bounds is None:
        raise ValueError("bounds must be provided when func has no 'bounds' attribute")
    X1, X2, Y = evaluate_grid(func, bounds, resolution)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X1,
                y=X2,
                z=Y,
                colorscale="Viridis",
                showscale=False,
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )
        ]
    )
    fig.update_layout(
        title=title or func.__name__,
        scene={"xaxis_title": "x1", "yaxis_title": "x2", "zaxis_title": "f(x)"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def plot_contour_with_trajectory(
    func: Callable[[np.ndarray], np.ndarray],
    result: OptimizationResult,
    bounds: Sequence[tuple[float, float]] | None = None,
    resolution: int = 200,
    title: str | None = None,
    max_frames: int = 150,
) -> go.Figure:
    """Animate an optimizer's best-so-far position over a contour of the landscape.

    Uses Plotly animation frames (play button + slider) built from
    ``result.position_history``. Long histories are subsampled to at most
    ``max_frames`` points, since each frame's trail is cumulative (O(n^2)
    data in the number of frames) and a run can have thousands of iterations.
    """
    bounds = bounds if bounds is not None else getattr(func, "bounds", None)
    if bounds is None:
        raise ValueError("bounds must be provided when func has no 'bounds' attribute")
    X1, X2, Y = evaluate_grid(func, bounds, resolution)
    full_path = np.array(result.position_history)
    frame_indices = np.unique(
        np.linspace(0, len(full_path) - 1, min(max_frames, len(full_path))).astype(int)
    )
    path = full_path[frame_indices]

    trajectory_trace = go.Scatter(
        x=path[:1, 0],
        y=path[:1, 1],
        mode="lines+markers",
        marker={"color": "red", "size": 6},
        line={"color": "red"},
    )

    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=path[: i + 1, 0],
                    y=path[: i + 1, 1],
                    mode="lines+markers",
                    marker={"color": "red", "size": 6},
                    line={"color": "red"},
                )
            ],
            traces=[1],
            name=str(i),
        )
        for i in range(len(path))
    ]

    fig = go.Figure(
        data=[
            go.Contour(x=X1[0], y=X2[:, 0], z=Y, colorscale="Viridis", showscale=False),
            trajectory_trace,
        ],
        frames=frames,
    )
    fig.update_layout(
        title=title or f"{func.__name__} — optimizer trajectory",
        xaxis_title="x1",
        yaxis_title="x2",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                            },
                        ],
                        "label": str(i),
                    }
                    for i in range(len(path))
                ]
            }
        ],
    )
    return fig


def render_search_animation_gif(
    func: Callable[[np.ndarray], np.ndarray],
    result: OptimizationResult,
    out_path: str,
    bounds: Sequence[tuple[float, float]] | None = None,
    resolution: int = 80,
    max_frames: int = 60,
    fps: int = 12,
    width: int = 700,
    height: int = 550,
    title: str | None = None,
) -> None:
    """Render a GIF of the search over a 3D surface: best-so-far trail on the surface, with the
    population projected as a 2D scatter on the floor plane beneath it.

    Unlike ``plot_contour_with_trajectory`` (interactive Plotly frames, for notebooks), this
    rasterizes each frame with kaleido and stitches them with imageio, since GitHub READMEs
    can only embed static images/GIFs, not interactive Plotly JS. Requires the ``viz``
    dependency group (``kaleido``, ``imageio``).
    """
    import imageio.v3 as iio

    bounds = bounds if bounds is not None else getattr(func, "bounds", None)
    if bounds is None:
        raise ValueError("bounds must be provided when func has no 'bounds' attribute")
    X1, X2, Y = evaluate_grid(func, bounds, resolution)

    best_path = np.array(result.position_history)
    populations = result.population_history
    num_steps = len(best_path)
    frame_indices = np.unique(
        np.linspace(0, num_steps - 1, min(max_frames, num_steps)).astype(int)
    )

    floor_z = Y.min() - 0.2 * (Y.max() - Y.min())
    surface = go.Surface(
        x=X1,
        y=X2,
        z=Y,
        colorscale="Viridis",
        showscale=False,
        opacity=0.85,
        contours={
            "z": {"show": True, "usecolormap": True, "project_z": True}
        },
    )
    scene = {
        "xaxis_title": "x1",
        "yaxis_title": "x2",
        "zaxis_title": "f(x)",
        "zaxis": {"range": [floor_z, Y.max()]},
        "aspectmode": "cube",
    }

    pngs = []
    for i in frame_indices:
        trail = best_path[: i + 1]
        data = [surface]
        if populations:
            pop = populations[i]
            best_xy = trail[-1]
            data.append(
                go.Scatter3d(
                    x=pop[:, 0],
                    y=pop[:, 1],
                    z=np.full(len(pop), floor_z),
                    mode="markers",
                    marker={"color": _POPULATION_COLOR, "size": 3},
                )
            )
            data.append(
                go.Scatter3d(
                    x=[best_xy[0]],
                    y=[best_xy[1]],
                    z=[floor_z],
                    mode="markers",
                    marker={"color": _BEST_COLOR, "size": 5},
                )
            )
        data.append(
            go.Scatter3d(
                x=trail[:, 0],
                y=trail[:, 1],
                z=func(trail.T),
                mode="lines+markers",
                line={"color": _BEST_COLOR, "width": 4},
                marker={
                    "color": _BEST_COLOR,
                    "size": [4] * (len(trail) - 1) + [7],
                },
            )
        )
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title or f"{func.__name__} — iteration {i}",
            scene=scene,
            showlegend=False,
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
        )
        pngs.append(iio.imread(fig.to_image(format="png", width=width, height=height, engine="kaleido")))

    iio.imwrite(out_path, pngs, duration=1000 / fps, loop=0)
