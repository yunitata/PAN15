# PAN15
Software submitted for PAN15 Author Verification (http://pan.webis.de/clef15/pan15-web/author-identification.html)<br />
The complete results can be seen here (http://www.tira.io/task/authorship-verification/) <br />
The details of the approach are explained in our notebook paper that can be accessed here http://www.uni-weimar.de/medien/webis/events/pan-15/pan15-papers-final/pan15-authorship-verification/sari15-notebook.pdf <br/>
We were submitting our system for three languages: dutch, english and spanish.

Before running the code, please make sure to install all dependencies software (sklearn).<br />
To train the software, type this following command in terminal:
```python
  python main.py -i $inputDataset -o $outputDir
```
$inputDataset is the the training dataset while $outputDir is the directory where the model will be saved <br />
and after training, you can use the model from the training step to predict the test data
```python
  python main.py -i $inputDataset -m $inputRun -o $outputDir
```
$inputDataset is the test data, $inputRun is the model while $outputDir is the directory for storing the result file. <br />
Please check the description of the task, to understand the input and output forms
