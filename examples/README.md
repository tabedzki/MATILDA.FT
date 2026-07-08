This folder contains several ready-to-go demonstration simulations that
highlight some of the features of MATILDA.FT.

Running an example
------------------

Every simulation is launched the same way, from inside the example's folder:

    ../../matilda.ft -in input

There is no longer a `-particle` or `-ft` command-line flag.  The kind of
simulation is selected by the first line of the input file:

  * `box ps ...`         -> particle-based simulation
  * `box fts scft|cl`    -> field-theoretic simulation

and the run itself is launched by a top-level command placed after `endBox`:

  * `run nvt <steps>`    -> particle-based run
  * `run vt <steps>`     -> field-theoretic run

You may also pass `-device <N>` to select a GPU.


PARTICLE-BASED MODELS
---------------------

coacervate:
Particle-based simulation of the system considered in Riggleman, Kumar, and
Fredrickson (J. Chem. Phys. 2012).  Charged polymers with smeared
electrostatics (enabled with the `doCharges` keyword and the
`potential charges` command).

dpd:
Spinodal demixing driven by Gaussian pair interactions with a DPD thermostat.
NOTE: the `potential dpd` command currently aborts during parsing due to a
source-side bug (see examples/dpd/README.txt); the input uses the correct
current syntax and will run once that bug is fixed.

dynamic-bonds:
Originally a dynamic-binding demo.  The dynamic-bonding feature (the old
`nlist ... bonding` / `extraforce ... lewis` commands) has been removed from
the code, so the input has been modernized as a best effort to run the same
particles as a plain Gaussian melt (see examples/dynamic-bonds/README.txt).

lamella:
A 3D simulation of an A-B diblock copolymer with f = 0.5 and large chi*N.  The
system is randomly initialized and is expected to form a defective lamellar
state.

model-A:
Particle-based implementation of the Gaussian-regularized Edwards model by
Villet and Fredrickson (J. Chem. Phys. 2014).


FIELD-THEORETIC SIMULATIONS
---------------------------

ft:
Calculates the equilibrium structure of an A-B homopolymer blend at chi*N = 3.5.
The system is initialized to a sine-wave potential profile.


OTHER
-----

extend:
A source-code extension tutorial (not a runnable input) showing how to add a
new group type to MATILDA.FT.  See extend/instructions.txt.  Note that the
class it modifies may have changed in the current source tree.
