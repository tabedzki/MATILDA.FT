This sample is set up to reproduce data from Riggleman, Kumar, Fredrickson,
JCP V136 024903 (2012). Specifically, it should be similar to data in Figure
4a, B=0.05, with E \approx 2700. 

The density of the condensed phases in [b^-3] units (as opposed to [Rg^-3]
units in the paper) should be approximately 6.6 or so.

Run with:

    ../../matilda.ft -in input

(The old "-particle" flag no longer exists; the box type is selected by the
"box ps" line inside the input file.  Charges are enabled with the "doCharges"
keyword and handled by the "potential charges <Bjerrum> <sigma>" line.)
