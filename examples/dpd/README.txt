Example of Dissipative Particle Dynamics (DPD).
Note that the Velocity Verlet (VV) integrator is used.
DPD requires a neighbor list ("neighbor_list all <rcut>").

Run with:

    ../../matilda.ft -in input

(The old "-particle" flag no longer exists; the box type is selected by the
"box ps" line inside the input file.)

KNOWN ISSUE (current source tree): the "potential dpd ..." command aborts
during input parsing.  ps_potentialDPD.cu reads its optional arguments with a
loop that leaves the input stringstream in a failed state, and the subsequent
ramp_check_input() call then dies with "failed to properly read".  This is a
source-side bug rather than stale input syntax; the input file here uses the
correct current DPD syntax and will run once that parser bug is fixed.
