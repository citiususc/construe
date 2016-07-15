# Beat Classification

This project includes an algorithm for automatic beat classification on ECG signals, described in the following paper:

 - T. Teijeiro, P. Félix, J.Presedo and D. Castro: *Heartbeat classification using abstract features from the abductive interpretation of the ECG*

The algorithm relies on the [abductive interpretation of an ECG record](README.md#interpreting-external-ecg-records) to obtain a set of qualitative morphological and rhythm features for each QRS observation in the interpretation result. Then, a clustering task provides a partition of the full set of QRS observations, and finally a label is assigned to each cluster, classifying all the beats in the record.

The classification stage is implemented in the `beat_classification.py` script, that receives as input a record in [MIT-BIH format](https://www.physionet.org/physiotools/wag/header-5.htm), a set of [annotations in the MIT format](https://www.physionet.org/physiotools/wag/annot-5.htm) with the interpretation results, and the clustering results, consisting of a simple tag assignment to each beat annotation in the interpretation results. As output, this script generates a copy of the input annotations file, with all beats labeled.

## Files required for the classification

In short, for the automatic classification of heartbeats using this algorithm, it is required:
 1. An ECG record in the [MIT-BIH format](https://www.physionet.org/physiotools/wag/header-5.htm) (Typically, a `.hea` file and a `.dat` file).
 2. A set of annotations resulting from the abductive interpretation of the record (The `.iatr` files included in the `classification_data/` directory).
 3. The results of the clustering task on the beat annotations (The `.cluster` files included in the `classification_data/` directory)

Since the abductive interpretation stage is a very time-consuming task, the annotation files with the results for all records in the [MIT-BIH Arrhythmia Database](https://www.physionet.org/physiobank/database/mitdb/) are provided in the `classification_data/` directory with the `.iatr` extension. These results can be obtained with the `record_processing.py` script, taking as initial evidence the `.atr` reference annotations in the database. Note that the interpretation results depend on the computing capabilities, so they may be slightly different from those provided in this repository.

With respect to the clustering results, unfortunately at this time we can not provide access to the implementation of the clustering algorithm. This algorithm is described in the following paper:

 - D. Castro, P. Félix, and J. Presedo: *A Method for Context-Based Adaptive QRS Clustering in Real Time*, IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 5, pp. 1660–1671, Sept 2015.

We have also included the clustering results for all records in the MIT-BIH Arrhythmia Database in the `classification_data/` directory, in a set of files with the `.cluster` extension. These files have a simple plain text format, with two columns of integer numbers. The first column contains the index of each beat within the interpretation annotation file, in 0-based indexing, and the second column contains the identifier of the cluster the beat belongs to.

## Using a customized clustering algorithm

If you want to use a different clustering strategy before the classification stage, you have to follow these steps:

 1. Get the annotation file resulting from the abductive interpretation of the record (or the already provided `.iatr` file).
 2. Apply your clustering algorithm to all annotations whose [NUM field](http://www.physionet.org/physiotools/wag/annot-5.htm#toc3) is one of the following: (1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 16, 25, 30, 34, 35, 38, 41).
 3. Generate a cluster file with the same format described in the previous section.

## Reproducing the results in the paper

To reproduce the classification results shown in the paper, you have to follow these steps:

 1. Download the [MIT-BIH Arrhythmia Database](https://www.physionet.org/physiobank/database/mitdb/) files to your hard disk (for example, in the `/tmp/mitdb/` directory).
 2. Copy the `.iatr` and `.cluster` files into the same directory.
 3. For all records to be tested, execute the `beat_classification.py` script: `python beat_classification.py -r /tmp/mitdb/$REC -a iatr -c cluster -o cls`. (Here $REC stands for the name of the record, and `.cls` is the selected extension for the annotations file with the classification results).
 4. For all records to be tested, compare the classification results with the manual reference using the [bxb application](https://www.physionet.org/physiotools/wag/bxb-1.htm): `bxb -r $REC -a atr cls -f 0 -t 0 -L file1 file2 2> /dev/null`.
 5. Get the aggregated statistics using the [sumstats application](https://www.physionet.org/physiotools/wag/sumsta-1.htm): `sumstats file1 > mitdb_validation`.

Finally, in the `mitdb_validation` you will find the full statistics of the classification results. Following we show the results for the full database:

|Record|Nn'|Sn'|Vn'|Fn'|On'|Ns|Ss|Vs|Fs'|Os'|Nv|Sv|Vv|Fv'|Ov'|No'|So'|Vo'|Fo'|Q Se|Q +P|V Se|V +P|S Se|S +P|RR err|
|------|---|---|---|---|---|--|--|--|---|---|--|--|--|---|---|---|---|---|---|----|----|----|----|----|----|-------|
|100|2239|0|0|0|0|0|33|0|0|0|0|0|1|0|0|0|0|0|0|100.00|100.00|100.00|100.00|100.00|100.00|2.23|
|101|1858|1|0|1|0|2|2|0|0|0|0|0|0|1|0|0|0|0|0|100.00|100.00|-|-|66.67|50.00|3.55|
|102|99|0|0|2084|0|0|0|0|0|0|0|0|4|0|0|0|0|0|0|100.00|100.00|100.00|100.00|-|-|17.32|
|103|2082|1|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|100.00|100.00|-|-|50.00|100.00|2.64|
|104|163|0|1|2063|0|0|0|0|0|0|0|0|0|1|0|0|0|1|0|99.96|100.00|0.00|-|-|-|29.69|
|105|2513|0|5|4|1|0|0|0|0|0|2|0|35|0|0|11|0|1|1|99.49|99.96|85.37|94.59|-|-|24.18|
|106|1493|0|17|0|0|0|0|0|0|0|14|0|490|0|0|0|0|13|0|99.36|100.00|94.23|97.22|-|-|109.08|
|107|0|0|3|2054|0|0|0|0|0|0|0|0|56|23|0|0|0|0|1|99.95|100.00|94.92|100.00|-|-|15.27|
|108|1685|1|0|2|0|2|4|5|0|0|48|0|11|0|0|4|0|1|0|99.72|100.00|64.71|18.64|80.00|36.36|80.13|
|109|2491|0|8|2|0|0|0|2|0|0|1|0|28|0|0|0|0|0|0|100.00|100.00|73.68|96.55|-|0.00|6.57|
|111|2122|0|0|0|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|100.00|100.00|100.00|50.00|-|-|9.67|
|112|2536|0|0|0|0|1|2|0|0|0|0|0|0|0|0|0|0|0|0|100.00|100.00|-|-|100.00|66.67|5.01|
|113|1788|0|0|0|0|0|0|0|0|0|1|6|0|0|0|0|0|0|0|100.00|100.00|-|0.00|0.00|-|5.70|
|114|1814|3|0|0|0|0|9|0|0|0|4|0|43|4|0|2|0|0|0|99.89|100.00|100.00|91.49|75.00|100.00|45.60|
|115|1951|0|0|0|0|0|0|0|0|0|2|0|0|0|0|0|0|0|0|100.00|100.00|-|0.00|-|-|2.75|
|116|2299|0|0|0|0|0|0|7|0|0|3|1|102|0|0|0|0|0|0|100.00|100.00|93.58|96.23|0.00|0.00|4.75|
|117|1534|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|100.00|100.00|-|-|100.00|100.00|7.88|
|118|2164|2|0|0|0|1|94|7|0|0|1|0|9|0|0|0|0|0|0|100.00|100.00|56.25|90.00|97.92|92.16|6.48|
|119|1540|0|0|0|0|0|0|0|0|0|0|0|442|0|0|3|0|2|0|99.75|100.00|99.55|100.00|-|-|101.90|
|121|1861|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|100.00|100.00|100.00|100.00|0.00|-|2.96|
|122|2475|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|99.96|100.00|-|-|-|-|24.00|
|123|1515|0|0|0|0|0|0|0|0|0|0|0|3|0|0|0|0|0|0|100.00|100.00|100.00|100.00|-|-|2.78|
|124|1526|34|0|3|0|3|2|0|0|0|2|0|46|2|0|0|0|1|0|99.94|100.00|97.87|95.83|5.56|40.00|45.33|
|200|1688|9|20|1|0|23|21|31|0|0|32|0|773|1|0|0|0|2|0|99.92|100.00|93.58|96.02|70.00|28.00|25.09|
|201|1612|40|0|0|0|2|76|0|0|0|1|20|197|2|0|10|2|1|0|99.34|100.00|99.49|90.37|55.07|97.44|152.12|
|202|2047|22|0|1|0|1|28|1|0|0|0|2|17|0|0|13|3|1|0|99.20|100.00|89.47|89.47|50.91|93.33|147.06|
|203|2278|1|44|4|1|35|0|4|0|1|207|1|384|1|2|9|0|12|0|99.30|99.87|86.49|64.65|0.00|0.00|84.47|
|205|2569|0|0|10|0|2|2|0|0|0|0|0|70|1|0|0|1|1|0|99.92|100.00|98.59|100.00|66.67|50.00|19.54|
|207|1446|1|7|0|0|5|106|2|0|0|90|0|201|0|0|2|0|0|0|99.89|100.00|95.71|69.07|99.07|93.81|14.32|
|208|1570|2|30|348|0|0|0|0|0|0|16|0|959|26|0|0|0|3|1|99.86|100.00|96.67|98.36|0.00|-|33.80|
|209|2602|20|0|0|0|18|362|0|0|0|1|0|1|0|0|0|1|0|0|99.97|100.00|100.00|50.00|94.52|95.26|13.26|
|210|2316|0|10|2|1|102|19|14|2|0|5|3|168|6|0|0|0|3|0|99.89|99.96|86.15|95.45|86.36|13.87|25.46|
|212|2748|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|100.00|100.00|-|-|-|-|5.93|
|213|2634|3|24|333|0|7|25|6|3|0|0|0|189|26|0|0|0|1|0|99.97|100.00|85.91|100.00|89.29|60.98|20.09|
|214|2003|0|2|3|0|0|0|0|0|0|0|0|254|0|0|0|0|0|0|100.00|100.00|99.22|100.00|-|-|8.99|
|215|3192|0|26|1|0|0|0|0|0|0|3|3|138|0|0|0|0|0|0|100.00|100.00|84.15|95.83|0.00|-|6.34|
|217|239|0|9|1800|0|0|0|0|0|0|5|0|142|1|0|0|0|11|1|99.46|100.00|87.65|96.60|-|-|79.05|
|219|2040|7|4|0|0|2|0|1|0|0|1|0|58|1|0|39|0|1|0|98.14|100.00|90.62|98.31|0.00|0.00|244.08|
|220|1947|2|0|0|0|5|92|0|0|0|0|0|0|0|0|2|0|0|0|99.90|100.00|-|-|97.87|94.85|59.27|
|221|2030|0|5|0|0|0|0|0|0|0|0|0|371|0|0|1|0|20|0|99.13|100.00|93.69|100.00|-|-|109.40|
|222|1921|141|0|0|0|134|275|0|0|0|0|0|0|0|0|7|5|0|0|99.52|100.00|-|-|65.32|67.24|118.49|
|223|2027|24|110|12|0|2|37|25|0|0|0|27|336|2|0|0|1|2|0|99.88|100.00|71.04|92.56|41.57|57.81|52.68|
|228|1684|1|1|0|0|4|2|10|0|0|0|0|351|0|0|0|0|0|0|100.00|100.00|96.96|100.00|66.67|12.50|20.04|
|230|2255|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|100.00|100.00|100.00|100.00|-|-|3.87|
|231|1568|1|0|0|0|0|0|0|0|0|0|0|2|0|0|0|0|0|0|100.00|100.00|100.00|100.00|0.00|-|5.37|
|232|389|5|0|0|0|2|1333|0|0|0|4|43|0|0|0|2|2|0|0|99.78|100.00|-|0.00|96.38|99.85|92.22|
|233|2216|0|1|3|0|1|7|1|0|0|1|0|821|8|0|12|0|8|0|99.35|100.00|98.80|99.88|100.00|77.78|66.92|
|234|2699|8|0|0|0|1|42|0|0|0|0|0|3|0|0|0|0|0|0|100.00|100.00|100.00|100.00|84.00|97.67|1.93|
|**Sum**|**89468**|**330**|**327**|**8731**|**3**|**355**|**2575**|**116**|**5**|**1**|**445**|**106**|**6708**|**106**|**2**|**118**|**15**|**85**|**4**||||||||
|**Gross**                      ||||||||||||||||||||**99.80**|**99.99**|**92.70**|**92.38**|**85.10**|**84.51**||
