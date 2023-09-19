# Intro  
AFG-PPIS is a novel framework for protein-protein interaction site prediction using multipy-head attention, deep graph convolution network and feed-forward neural network.   
![AFG-PPIS_framework](https://github.com/fxh1001/AGF-PPIS/IMG/AGF-PPIS.png)  

# System requirement  
AFG-PPIS is developed under Linux environment with:  
python  3.7.13  
numpy  1.19.2  
pandas  1.3.5  
torch  1.13.0 

# Software and database requirement  
You need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)   

# Build database and set path  
1. Use `makeblastdb` in BLAST+ to build UniRef90 ([guide](https://www.ncbi.nlm.nih.gov/books/NBK569841/)).  
2. Build Uniclust30 following [this guide](https://github.com/soedinglab/uniclust-pipeline).  
3. Set path variables `UR90`, `HHDB`, `PSIBLAST`, `HHBLITS` and `DSSP` in `get_feature.py`.  

# Run AFG-PPIS for prediction  
For a protein chain in PDB:  
```
$cd AFG-PPIS
```

```
$python prediction.py --querypath ../AFG-PPIS --filename 1acb.pdb --chainid I --cpu 10
```


Contact:  
Xiuhao Fu (fxh1001@hainanu.edu.cn)

Feifei Cui (fefeicui@hainanu.edu.cn)

Zilong Zhang (zhangzilong@hainanu.edu.cn)

