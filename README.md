# AIDE_Reproduction
[AIDE (Annotation efficient deep learning for automatic medical image segmentation)](https://doi.org/10.1038/s41467-021-26216-9) is an open-source framework to handle imperfect training datasets.
### Data preparation
- If you want to run the code, you need to download the [CHAOS](https://chaos.grand-challenge.org/) datasets for tasks.

- Data should be stored in the correct directory tree.

For CHAOS, it should like this:

  $inputs_chaos

  |-- All_Sets  
  |--|--Case_No      
  |--|--|--T1DUAL         
  |--|--|--|--DICOM_anon            
  |--|--|--|--Ground

### Parameters
Given that I only have a laptop with a 3060 GPU to train, I changed the following parameters: **"batch_size = 2", "num_workers = 0".**

### Result
I trained 100 epoches with the pace of 30 sec per epoch, and the result are as follows:

| Patient_case |   Dice   |    IoU    |    TP    |     TN     |    FP    |    FN    |
|--------------|----------|-----------|----------|------------|----------|----------|
|      2       | 0.8396   | 0.7236    | 100287   | 2017948    | 29216    | 9093     |
|      5       | 0.5954   | 0.4239    | 31072    | 1892783    | 688      | 41537    |
|     10       | 0.3742   | 0.2302    | 14818    | 3212424    | 86       | 49472    |
|     13       | 0.9042   | 0.8252    | 64402    | 2410276    | 6125     | 7517     |
|     19       | 0.9398   | 0.8864    | 94452    | 2381769    | 4074     | 8025     |
|     20       | 0.8848   | 0.7934    | 67128    | 2071939    | 6012     | 11465    |
|     31       | 0.7640   | 0.6181    | 27891    | 1920957    | 1852     | 15380    |
|     33       | 0.0      | 0.0       | 0        | 2031630    | 81802    | 49256    |
|     34       | 0.6051   | 0.4338    | 35760    | 2211322    | 892      | 45786    |
|     39       | 0.7041   | 0.5433    | 23515    | 1660658    | 822      | 18941    |
