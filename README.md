# Glioblastoma.Prognosis
Glioblastoma Multiforme Prognosis: MRI Missing Modality Generation, Segmentation and Radiogenomic Survival Prediction<br />
Dataset: BraTS2020


<br />
<div align="center">
  <h3 align="left">Step 1: Generating T2 MRI Missing Modality</h3>
  
  <a href="">
    <img src="https://github.com/AmirMz1/Glioblastoma.Prognosis/blob/main/real_image.png?raw=true" width="1920px"/>
  </a>
  <br />
  <a href="">
    <img src="https://github.com/AmirMz1/Glioblastoma.Prognosis/blob/main/generated_image.png?raw=true" width="1920px"/>
  </a>

</div>

<br />
<div align="center">
  <h3 align="left">Step 2: Segmentation</h3>

  
  <h4 align="center">True Mask</h4>
  <a href="">
    <img src="https://github.com/AmirMz1/Glioblastoma.Prognosis/blob/main/true_mask.png?raw=true" width="1080px"/>
  </a>
  <br />
  <h4 align="center">Segmented Mask</h4>
  <a href="">
    <img src="https://github.com/AmirMz1/Glioblastoma.Prognosis/blob/main/segmentation_results.png?raw=true" width="1080px"/>
  </a>


</div>


<br />
<div align="center">
  <h3 align="left">Step 3: Extracting radiomics data from sgemented reigons</h3>
  <br />
  <br />

  <h3 align="left">Step 4: Combining the radiomics, clinical and CNV data</h3>
  <h3 align="left">Step 5: Train an ANN to predict patient state as 3 classes (high risk, mid risk, low risk)</h3>

  <a href="">
    <img src="https://github.com/AmirMz1/Glioblastoma.Prognosis/blob/main/traning%20and%20Validation%20Accuracy.png?raw=true" width="1080px"/>
  </a>
  

</div>
