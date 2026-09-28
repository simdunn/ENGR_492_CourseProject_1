##choose mesh size, use find load position.
#once converges, program compares mesh with previous one
#after converges, check forces, compare to analitical answer and print graphs

#mesure CPU work in seconds
from time import process_time

import numpy as np
import matplotlib.pyplot as plt

#import funcs from function file
from functions import solve_fem, find_load_positions

#Inputs
L=1000.0 #shaft L (mm)
D=750.0 #left diameter (mm)
d=250.0 #right diameter (mm)
E=260000.0 #Youngs modulus N/mm^2
P=500.0 #Applied force (N) + is too the right

#Actual areas in mm^2
A_left = np.pi * D**2 /4
A_right = np.pi * d**2 / 4

#Chosen tolerances (we dabble in precision)
stress_tolerance= 1e-6
position_tolerance= 1e-5
mesh_tolerance= 1e-4 #change in x and max displacement 

#solve for load position search usimng these element counts
mesh_sizes = [10, 20, 40, 80, 160, 320]

#function to run both functions with inputs above

def main():

    #record starting cpu time to subract from finish time
    start = process_time()

    # list that stores a row of results per mesh
    history = []

    #false until mesh converges
    converged = False

    # :g is a number display, not to be confused with "g", im in too deep to go back and change it 

    print(f"Initial search interval: {0.1*L} mm to {0.9*L} mm")
    print(f"Required relative stress tolerance: {stress_tolerance:g}")
    print(f"Required search interval width: < {position_tolerance:g} mm")
    print(f"Required mesh change in x and max u: < {100*mesh_tolerance:g} %\n")
    print(f"Elements      x(mm)     max u(mm)     mesh change (%)")

    #search returns chosen position and fem data and u from that position

    for n_elements in mesh_sizes:
        x_load, result = find_load_positions(n_elements, L, D, d, E, P, stress_tolerance, position_tolerance)

        #Takes magnitude and picks largest (mm)
        u_max = np.max(np.abs(result["u"]))
        mesh_change = np.nan #

        #if first mesh has no previous results
        if len(history) > 0:
            #[-1]: last row, [1:3]: columns 1 and 2 (x and u_max)
            previous_x, previous_u = history[-1][1:3]

            #calculate mesh change in x and u_max
            change_x = abs(x_load - previous_x) / abs(previous_x)
            change_u = abs(u_max - previous_u) / u_max

            #changes must be small 
            mesh_change = max(change_x, change_u)

        # Re calc final stress miss match
        mismatch = abs(result["g"]) / np.max(np.abs(result["end_stress"]))

        #Add a row to history

        history.append([n_elements, x_load, u_max, 100*mesh_change, 100*mismatch])

        #print table
        print(
            f"{n_elements:8d}  {x_load:11.6f}  "
            f"{u_max:13.6f}  {100*mesh_change:15.6f}  "
        )

        if mesh_change < mesh_tolerance:
            converged = True
            break #break when converged
    
    #if not converged none of the meshes met the tolerance
    if not converged:
        raise RuntimeError("Mesh has not converged. add finer mesh sizes ")
    
    #Check equilibrium and compare with analytical result
    R= result["reactions"] 

    #R[0] is left support reaction, R[-1] is right support reaction
    #force balance Rleft + Rright +P = 0 
    #Equilibrium error is / P
    equilibrium_error = abs(R[0] + R[-1] + P) / abs(P)

    #at each free node, unbalanced force should be zero, so we can check the internal forces at each node
    free_node_error = np.max(np.abs(R[1:-1])) / abs(P)

    if max(equilibrium_error, free_node_error) > 1e-6:
        raise RuntimeError("Equilibrium check failed. Check the FEM.")
    

    #VERIFICATION

    x_reference = L * d / (D + d)  # Analytical solution for load position

    #common stress magnitude s, N is internal forces
    stress_reference = P / (A_left + A_right)  # Analytical solution for stress magnitude

    N_left = stress_reference * A_left
    N_right = -stress_reference * A_right   

    #Linear diameter equation at load location
    diameter_at_load = D + (d-D) * x_reference / L

    #integrating N/(E*A(z)) for the left gives load displacement
    u_reference = (
        4 * N_left * x_reference 
        / (np.pi * E * D * diameter_at_load)
    )

    #errors
    x_error = 100 * abs(x_load - x_reference) / x_reference
    u_error = 100 * abs(u_max - u_reference) / u_reference

    #CPU time
    analysis_cpu = process_time() - start
    # summary is a string of printed results. \n in a string starts a new line.
    summary = (
        f"\nConverged mesh: {n_elements} elements\n"
        f"Load position from the LARGE left end: {x_load:.6f} mm\n"
        f"End stresses, left/right: {result['end_stress']} MPa\n"
        f"Reactions, left/right: {R[0]:.6f}, {R[-1]:.6f} N\n"
        f"Maximum displacement: {u_max:.9e} mm\n"
        f"End-stress mismatch: {100*mismatch:.6e}%\n"
        f"Relative equilibrium error: {equilibrium_error:.3e}\n"
        f"Maximum relative free-node force residual: {free_node_error:.3e}\n"
        f"Reference x: {x_reference:.6f} mm; FEM error: {x_error:.6f}%\n"
        f"Reference max u: {u_reference:.9e} mm; FEM error: {u_error:.6f}%\n"
        f"Analysis CPU time: {analysis_cpu:.3f} s (computer-dependent)\n"

    )
    print(summary)

    #Node position on the converged mesh
    z=result["z"]

    #Plotting
    #
    trial_positions = np.linspace(0.1*L, 0.9*L, 41)
    trial_g = [] #one stress difference for each trial position

    for positions in trial_positions:
        trial_result = solve_fem(positions, n_elements, L, D, d, E, P)
        trial_g.append(trial_result["g"])

    #fig is figure window
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        constrained_layout=True
    )

    #ax selects currenr panel
    ax = axes[0, 0]

    #convert Mpa to Kpa 
    ax.plot(
        trial_positions,
        1000*np.array(trial_g),
        label="FEM g(x)"
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.plot(
        #ro is red circle
        x_load, 1000*result["g"], "ro", label="Selected position"
    )
    ax.set(
        xlabel="Load position x (mm)",
        ylabel="Stress difference g(x) (kPa)",
        title="1. Equal end stresses search",

    )
    ax = axes[0, 1]
    h = np.array(history) #convert rows to a table

    #h[:, 0] all rows of column 0,
    # h[:, 1] all rows of column 1, (seleced load position)
    # "o-" is a line with circles at the data points

    ax.plot(
        h[:, 0], h[:, 1], "o-", label="FEM"
    )
    ax.axhline(x_reference, color="black", linestyle="--", label="Analytical reference")

    ax.set(
        xlabel="Number of elements",
        ylabel="Load position x (mm)",
        title="2. Load position convergence",
    )

    #node diameter is an array of diameters at each node position
    node_diameter = D + (d-D) * z / L

    #where uses left expression b4 load and right after 
    u_exact= np.where(
        z<= x_reference,
        4 * N_left * z / (np.pi * E * D * node_diameter),
        -4 * N_right * (L-z) / (np.pi * E * d * node_diameter)
    )
    ax = axes[1, 0]

    #convert mm to nm 
    ax.plot(z, result["u"] * 1e6, label="FEM")
    ax.plot(z, u_exact * 1e6, "k--", label="Analytical reference")
    ax.set(
        xlabel="Position along shaft (mm)",
        ylabel="Displacement (nm)",
        title="3. Axial displacement"
    )
    ax = axes[1, 1]

    #Stairs show each elements const stress
    ax.stairs(
        1000*result["stress"],
        z,
        label="FEM Element stress"
    )

    #Plot the 2 analytical stress curves sepertely 
    #point load makes N jump
    for positions, force in [
        (np.linspace(0, x_reference, 120), N_left),
        (np.linspace(x_reference, L, 120), N_right)
    ]:
        area = np.pi * (D + (d - D) * positions / L)**2 / 4
        ax.plot(positions, 1000 * force / area, "k--")

    # Empty data adds a legend entry for the two dashed analytical curves
    ax.plot([], [], "k--", label="Analytical reference")

    # Mark the physical-end stresses recovered using the actual end areas.
    ax.plot(
        [0, L],
        1000 * result["end_stress"],
        "ro",
        label="Recovered end stresses"
    )    

    ax.set(
        xlabel="Position along shaft (mm)",
        ylabel="Axial stress (kPa)",
        title="4. Tension(+), compression(-) "
    )

    #add grids and legends to all panels
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.show()

    #wasnt working but apparently this line fixes it? thx reddit
if __name__ == "__main__":
    main()




    



     




