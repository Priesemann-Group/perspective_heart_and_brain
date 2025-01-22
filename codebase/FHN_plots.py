import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

projectfolder = "../figures/"
simulation_name = "default"
mpl.rcParams.update({"font.size": 16})



def plot_graph_on_circle(
    y,
    J,
    ax,
    subset_size=50,
    format_axes=True,
    plot_connections=True,
    arrows=True,
    marker_size=200,
    idle_color="white",
    spiking_color="royalblue",
    weighted_connections_cmap=None,
    weighted_connections_norm=None,
):
    """
    Plot the graph on a circle with nodes colored based on their V value.
    Parameters:
        J (scipy.sparse matrix): adjacency matrix
        y (numpy.ndarray): (N) states of the neurons
        subset_size: number of nodes to plot, if 'all' then subset_size=N
    """
    J_dense = J.todense()
    #else:
   
    N = len(y)

    # Select a subset of nodes to plot
    if type(subset_size) == str and subset_size == "all":
        subset_size = N
    subset_indices = np.random.choice(N, subset_size, replace=False)

    # Create positions for the nodes on a circle
    theta = np.linspace(0, 2 * np.pi, subset_size, endpoint=False)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))

    if format_axes:
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

    # Plot the connections of the subset
    if plot_connections:
        i_indices, j_indices = np.meshgrid(
            subset_indices, subset_indices, indexing="ij"
        )
        mask = J_dense[i_indices, j_indices] != 0
        i_indices, j_indices = i_indices[mask], j_indices[mask]

        for i, j in zip(i_indices, j_indices):
            if type(weighted_connections_cmap) == type(None):
                color = "black"
            else:
                color = weighted_connections_cmap(
                    weighted_connections_norm(J_dense[i, j])
                )
            if arrows:
                ax.arrow(
                    x=positions[i % subset_size, 0],
                    y=positions[i % subset_size, 1],
                    dx=positions[j % subset_size, 0] - positions[i % subset_size, 0],
                    dy=positions[j % subset_size, 1] - positions[i % subset_size, 1],
                    color=color,
                    alpha=1,
                    head_width=0.02,
                    length_includes_head=True,
                )
            else:
                ax.plot(
                    [positions[i % subset_size, 0], positions[j % subset_size, 0]],
                    [positions[i % subset_size, 1], positions[j % subset_size, 1]],
                    "-",
                    color=color,
                    alpha=0.1,
                )
    # Add a colorbar with specific ticks
    subset_y = y[subset_indices]
    
    norm = plt.Normalize(vmin=-np.max(np.abs(subset_y)), vmax=np.max(np.abs(subset_y)))
    cmap = plt.get_cmap("seismic")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('V', rotation=0, labelpad=15, fontsize=16, color='black')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=14)
    colors = cmap(norm(subset_y))
    ax.scatter(
        positions[:, 0], positions[:, 1], c=colors, edgecolors="black", s=marker_size
    )

def plot_kymograph(
    fig,
    ax,
    model,
    t_start=100,
    cmap="seismic",
    xlabel="time (a.u.)",
    ylabel="Node",
    cbar_axes=None,
    cbar_shrink=0.6,
    cbar_location="top",
    cbar_orientation="horizontal",
):
    # Plot the Kymograph
    im=ax.imshow(
        model.vs[t_start:, ::-1].T,
        extent=(np.min(model.ts[t_start:]), np.max(model.ts[t_start:]), 0, model.N),
        aspect="auto",
        cmap=cmap,
        interpolation="none",
        vmin=-np.max(np.abs(model.vs)),
        vmax=np.max(np.abs(model.vs))
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # plot colorbar
    if type(cbar_axes) == type(None):
        cbar_axes = ax
    cbar = fig.colorbar(
        im,
        ax=cbar_axes,
        location=cbar_location,
        shrink=cbar_shrink,
        orientation=cbar_orientation,
    )
    plt.show()
    return im, cbar

def heart_plot(
        fig,
        ax,
        model,
        t_frame=100,
        cbar_axes=None,
        cbar_shrink=0.6,
        cbar_location="right",
        cbar_orientation="vertical",
        cmap='seismic'

  ):


    array=model.vs.T
# Display the first frame
    array=array.reshape(model.N,model.N,-1)
    img = ax.imshow(array[:,:, t_frame], cmap=cmap, interpolation="bilinear", vmin=-np.max(np.abs(array)), vmax=np.max(np.abs(array)))
	
    
# Add a colorbar with specific ticks
    if type(cbar_axes) == type(None):
        cbar_axes = ax
    cbar = fig.colorbar(
        img,
        ax=cbar_axes,
        location=cbar_location,
        shrink=cbar_shrink,
        orientation=cbar_orientation,
    )
    ax.set_aspect('auto')
    cbar.set_label('V', rotation=0, labelpad=15, fontsize=12, color='black')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  
    plt.show()