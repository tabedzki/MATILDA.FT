Simple example of dynamic binding (with optional induced charge).

IMPORTANT: The dynamic-bonding feature this example was originally written for
-- the old "nlist ... bonding <donor/acceptor map>" and
"extraforce ... lewis ..." commands -- is NOT present in the current
MATILDA.FT source tree.  As a result the original dynamic-binding simulation
(and the bonds_out connectivity output described below) cannot be reproduced.

The input file in this folder has been modernized to the current input format
as a best effort: it runs the same 10-particle system as a plain soft-sphere
(Gaussian) melt so that the example still exercises the current parser and
runs to completion.  The dynamic-bond / induced-charge physics is not
included because the feature has been removed from the code.

Run with:

    ../../matilda.ft -in input

--------------------------------------------------------------------------
Original description (for reference; requires the removed dynamic-bond code):

Output file (bonds_out) describes the connectivity within the system.

Sample from the output file:

# TIMESTEP, no. of bonds, no. of free bonds, no. of total possible bonds
# List of bond partners (donor particle id - acceptor particle id)

TIMESTEP: 800 2 3 5
8 7
10 9
