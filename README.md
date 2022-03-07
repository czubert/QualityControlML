# QualityControlML
Program for quality control of SERSitive products

Where the idea came from

Application responsible for training estimators based on data available
Program will be used as the Quality Control of SERSitive substrates based on the spectra.
There are two types of data you can acquire using SERS substrate:
- the background of the active surface area
- SERS "fingerprint" of the measured compound, in our case it was 4-paramercaptobenzoic acid (PMBA),
which we use to find out if our substrates have all the parameters (reproducibility, homogeneity and enhancement)
at expected rate. This is our Quality Control, which is not the best as:
- as we need to use at least one substrate from each batch (15 pcs) - cost-prohibitive
- we measure the background spectra after production process and then we immerse it in PMBA for 20h - time-consuming
- we estimate the quality of the whole batch basing on 1-2 substrates that were immersed - low accuracy
We found out that there is a dependence between the background spectra and the quality of the substrates.
Therefore, the idea of this program was to check if the dependence is true and if we can estimate the quality
of the substrate from the background spectrum.
