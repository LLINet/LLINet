# Look, Listen and Infer

This repo is pytorch implement for our paper "Look, Listen and Infer" published in ACM Multimedia 2020. More details and results can be seen in the [project page](https://LLINet.github.io).



**illustration of our work:**

<img src=".\img\pipeline.jpg" alt="pipeline" style="zoom:25%;" />



## Dataset

1. We build new dataset named **INSTRUMENT_32CLASS** to train and evaluate our model,  of which training set contains 24 classes instruments and test set contains 8 classes. The classes are:

   + Training set: 

     > Accordion, Bagpipes, Banjo, Bell, Cello, Chime, Clarinet, Drum, Flute, Gong, Guitar, Harmonica, Harp, Mandolin, Marimba, Piano, Saxophone, Singing bowl, Sitar, Steel pan, Trumpet, Ukulele, Violin, Zither

   + Test set:

     > Didgeridoo, Double bass, Electric guitar, French horn, Glockenspiel, Shofar, Steel guitar, Tabla

2. There are 3604 image-audio-segmentation groups extracting from videos. The samples and the distribution of dataset shows as following:

   <img src=".\img\dataset.jpg" alt="dataset" style="zoom:50%;" />

3. You can download **INSTRUMENT_32CLASS** from [there](https://drive.google.com/file/d/1O193VHG5FAmt8XRLyB4bZWyWE-KQoAQn/view?usp=sharing), and the directory structure shows as following:

  ```
  Dataset_32classes
  |——traindata
  |	|——train
  |	|	|——Accordion
  |	|	|	|——sound
  |	|	|	|	|——Accordion00001.wav
  |	|	|	|	|——Accordion00002.wav
  |	|	|	|	|——...
  |	|	|	|——video
  |	|	|	|	|——Accordion00001.jpg
  |	|	|	|	|——Accordion00002.jpg
  |	|	|	|	|——...
  |	|	|——Bagpipes
  |	|	|	|——sound
  |	|	|	|	|——Bagpipes.wav
  |	|	|	|	|——...
  |	|	|	|——video
  |	|	|	|	|——Bagpipes.jpg
  |	|	|	|	|——...
  |	|	|——...
  |	|——test
  |	|	|——Didggeridoo
  |	|	|	|——sound
  |	|	|	|	|——Didggeridoo00001.wav
  |	|	|	|	|——...
  |	|	|	|——video
  |	|	|	|	|——Didggeridoo00001.jpg
  |	|	|	|	|——...
  |	|	|	|——segment
  |	|	|	|	|——Didggeridoo00001.jpg
  |	|	|	|	|——...
  |	|——test_without_seg
  |	|	|——Didggeridoo
  |	|	|	|——sound
  |	|	|	|	|——Didggeridoo00001.wav
  |	|	|	|	|——...
  |	|	|	|——video
  |	|	|	|	|——Didggeridoo00001.jpg
  |	|	|	|	|——...
  |	|	|——...
  |	|——test_audio_files.pickle
  |	|——test_image_files.pickle
  |	|——test_class_ids.pickle
  |	|——train_audio_files.pickle
  |	|——train_image_files.pickle
  |	|——train_class_ids.pickle
  
  ```

​		*the `test` folder contains cropped images for the sake of sound localization. And the `test_without_seg` contains original images used for retrieval and recognition.

4. On the other way, you can build your own dataset by:

   **Step 1 :** Arrange directory structure mentioned above 

   **Step 2 :** Fill folders with your datas

   **Step 3 :** Run the `preprocessing_datafile.py` in `Dataset_32classes` folder to generate index files automatically：

   ```powershell
   python preprocessing_datafile.py
   ```

   Generated files are used for loading datas. Where

   + `test/train_audio_files.pickle` includes the path of audio files in test/train set.
   + `test/train_image_files.pickle` includes the path of image files in test/train set.
   + `test/train_class_ids.pickle` includes the class ids of instances in test/train set.



## **Training**

   + **Environment :** `Ubuntu 16.04 LTS` , `Python` == 3.6.9, `PyTorch` == 1.3.0, `librosa` == 0.7.1

   + **Training in different task :** Use `--mode` to select training task,   `--eva` to decide whether evaluate after every epoch, `--gamma_att` to control weight of attention loss. For example:

     ```powershell
     #sound localiztion with 0.8 attention loss
     python run.py --mode sl --gamma_att 0.8
     #retrieval without evaluation
     python run.py --mode retrieval --eva no
     #zsl with default setting
     python run.py --mode zsl
     ```

   + **Other Hyperparameters :** can be referred in `run.py` , e.g.  number of epochs `--n_epochs`, size of batch `--batch_size`.



## Cite

```latex
@inproceedings{jia2020look,
  title={Look, Listen and Infer},
  author={Jia, Ruijian and Wang, Xinsheng and Pang, Shanmin and Zhu, Jihua and Xue, Jianru},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3911--3919},
  year={2020}
}
```

