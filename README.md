# ECE-147-Project

### Performance Quicklook

| model name     | input                   | epoch | train acc | valid acc | num params  |  test acc  |
|----------------|-------------------------|-------|-----------|-----------|-------------|------------|
| LSTM           | raw                     | 30    | 0.9       | 0.31      | 40K         |   	  |
| RNN            | raw                     | 30    | 0.87      | 0.27      | 14K         |   	  |
| ConvLSTM       | raw                     | 30    | 0.86      | 0.36      | 58K         |   	  |
| DeepConvNet    | raw                     | 87    | 0.96      | 0.68      | 27K         | 0.64   	  |
| ShallowConvNet | raw                     | 30    | 0.83      | 0.57      | 46K         | 0.55   	  |
| ConvLSTM       | max, avg, subsampling   | 30    | 0.9       | 0.66      | 27K         |    	  |
| ATCNet         | raw                     | 30    | 0.83      | 0.67      | 123K        | 0.67   	  |
| ViT            | raw                     | 30    | 0.26      | 0.25      | 628K        |   	  |
| EEGNet_Modified| raw                     | 200   | 0.95      | 0.72      | 17.7K       | 0.73   	  |
| EEGNet_Modified| trim                    | 200   | 0.93      | 0.74      | 17.7K       | 0.74   	  |

### How to replicate the result

1. Create a conda virtual environment with requirements.txt
2. Put EEG data in a folder called “EEG_data”, then move the folder into “./EEG” directory
3. Inside “./EEG/src/main.py”, change the “wb_logger” to use your wandb account
4. Inside “./EEG” directory, run 


python ./src/main.py --data_dir ./EEG_data/ --model_name EEGNet_Modified


5. After 200 epochs (this is set by default), the last checkpoint should reach an accuracy of 72.9%