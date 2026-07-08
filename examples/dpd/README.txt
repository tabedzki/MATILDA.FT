Example of Dissipative Particle Dynamics (DPD).
Note that the Velocity Verlet (VV) integrator is used.
DPD requires a neighbor list ("neighbor_list all <rcut>").

Run with:

    ../../matilda.ft -in input

(The old "-particle" flag no longer exists; the box type is selected by the
"box ps" line inside the input file.)

KNOWN ISSUE: on source trees without the fix for issue #21 (PR #24), the
"potential dpd ..." command aborts during input parsing with "failed to
properly read" — a source-side parser bug, not stale input syntax.  The input
file here uses the correct current DPD syntax and runs once that fix is in.
