# Anomary Detection for time series data using LSTM model
Anomaly detection using LSTM model and prediction error distribution.
The original paper is [[1]] and this implementation uses [[2]] as a reference.
For the training and the test, we use ECG dataset [[3]].
The model used in [[1]] and [[2]] contains 2 LSTM layers
but our model contains only 1 LSTM layer.

## Requirements
* numpy==1.15.4
* matplotlib==3.0.0
* pandas==0.18.1
* sklearn==0.20.1
* Keras==2.2.4

## Usage
We assume the current directory contains the input data directory
data/. Enter the following command.
```console
$ python error_vectors_fitting.py
```
## References
[[1]]  Malhotra, Pankaj, et al. "Long short term memory networks for anomaly detection in time series." Proceedings. Presses universitaires de Louvain, 2015.

[[2]] LSTM for Anomaly Detection in time series. GRID INC..
https://www.renom.jp/notebooks/tutorial/time_series/lstm-anomalydetection/notebook.html

[[3]] http://www.cs.ucr.edu/~eamonn/discords/

[1]: https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf
[2]: https://www.renom.jp/ja/notebooks/tutorial/time_series/lstm-anomalydetection/notebook.html
[3]: http://www.cs.ucr.edu/~eamonn/discords/