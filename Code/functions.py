import numpy as np

#z is the node positions along the shaft, u is the displacement at each node, stress is the axial stress in each element, reactions are the reaction forces at the fixed ends, end_stress are the stresses at 
# the ends of the shaft, and g is the difference in magnitude of the end stresses.

#FEM calculation for one load position (trial position) 
def solve_fem(x_kload, n_elements, L, D, d, E, P):

#build meshes and matricies, solve for u, then stresses
#z= node position


    #Check so point load stays within the shaft
    if not (0 <= x_kload <= L):
        raise ValueError("Load position must be within the shaft length.")

    #check if elements is  even and 2 or more
    if n_elements < 2 or n_elements % 2 != 0:
        raise ValueError("Number of elements must be an even number and at least 2.")

    #Areas
    A_left = np.pi * (D**2) / 4
    A_right = np.pi * (d**2) / 4

    ##Half of the elements on each side of the shaft, so the load stays on the node
    n_half = n_elements // 2

    #make nodes equal spaced along the shaft
    left_nodes = np.linspace(0, x_kload, n_half + 1)
    right_nodes = np.linspace(x_kload, L, n_half + 1)[1:]  # Exclude the first node to avoid duplication
    z = np.concatenate([left_nodes, right_nodes])
    load_node= n_half

    ##Elements with zeros (e = element)
    Le=np.zeros(n_elements)
    A=np.zeros(n_elements)
    k=np.zeros(n_elements)
    K=np.zeros((n_elements+1,n_elements+1))
    F=np.zeros(n_elements+1)

    for e in range(n_elements): #i and j are left and right nodes
        i = e
        j = e + 1
        Le[e] = z[j] - z[i]
        z_mid = (z[i] + z[j]) / 2

        #Slope of the shaft in terms of radius
        r=0.5 * (D + (d-D) * z_mid / L)
        A[e] = np.pi * r**2


        # delta = PL/AE 
        #rearanging gives P = k* delta, where k = AE/L
        k[e] = E * A[e] / Le[e]

        #2-node stiffness matrix
        ke=np.array([[k[e], -k[e]], [-k[e], k[e]]])

        #Add each element at its global node position
        #shared nodes recive contributions fromm both
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[i, i] += ke[0, 0]
        K[j, j] += ke[1, 1]

    F[load_node] = P #P at load node
    #stores one u per node
    u=np.zeros(n_elements+1) 

    #Both ends are fixed 
    #using partitioning method to solve for displacements
    K_reduced = K[1:-1, 1:-1]
    F_reduced = F[1:-1]

    # Solve for unknown u values so that K_reduced @ u = F_reduced
    u[1:-1] = np.linalg.solve(K_reduced, F_reduced)


    #use calculated u values to get reaction forces at the fixed ends
    #reactions[0] and reactions[-1] are the support reactions @ nodes
    reactions = K @ u - F

    # delta = u_j - u_i, force = k * delta
    #normal stress is then internal force divided by disk area
    # P is not internal force 

    stress = np.zeros(n_elements)
    for e in range(n_elements):
        delta = u[e + 1] - u[e]
        axial_force = k[e] * delta
        stress[e] = axial_force / A[e]

    end_stress = np.array([-reactions[0] / A_left, reactions[-1] / A_right])

#g is the difference in sigma
#g>0 left has larger stress
#g<0 right has higher 
#g=0 2 magnitudes match (what we are looking for )
    g = abs(end_stress[0]) - abs(end_stress[1])

    #Legend for printing outputs:
    return {"z": z, "u": u, "stress": stress, "reactions": reactions, "end_stress": end_stress, "g": g}


#Find load positions where g is aprox 0

def find_load_positions(n_elements, L, D, d, E, P, stress_tolerance, position_tolerance):
    a, b = 0.1 * L, 0.9 * L  # Start with a range that avoids the very ends of the shaft
#finds the position where g is close to zero, meaning the end stresses are equal in magnitude
    g_a = solve_fem(a, n_elements, L, D, d, E, P)["g"]
    g_b = solve_fem(b, n_elements, L, D, d, E, P)["g"]

    #product of g_a and g_b should be negative for a root to exist in the interval of the shaft
    if g_a * g_b > 0:
        raise ValueError("interval does not bracket a root. Please choose a different interval.")


    for iteration in range(80):  # Limit iterations to prevent infinite loops
        x_load = (a + b) / 2
        result = solve_fem(x_load, n_elements, L, D, d, E, P)
        g_mid = result["g"]

        mismatch = abs(g_mid) / np.max(np.abs(result["end_stress"]))

        if mismatch < stress_tolerance and (b - a)  < position_tolerance:
            return x_load, result

        if g_a * g_mid > 0:
            # Same signs: the root is in the other half of the interval.
            a = x_load
            g_a = g_mid
        else:
            b = x_load
    #if all 80 attemps fail
    raise RuntimeError("Bisection did not meet the chosen tolerances.")
   
