# This is a masters thesis on Lie/Deception Detection using computer vision and deep learning algorithms for the degree of MSc. Computational and Software techniques in Engineering at Cranfield University, United Kingdom.


# Before Running the Codes:

  - Train.py: For training the datasets.
  - Test.py: For Deception detection and detecting micro-expressions.
  - Emotions.h5: The model gotten after running train.py

# Please Remember to change the following directories:

Test.py :

  - Line 7: Please change the path to where you must have saved the haar-cascade file in the folder.
  - Line 9: Please change the path to where you must have saved the model(emotions.h5) file in the folder.
  - lines 14 & 15: Please change the path to where you must have saved the videos submitted in the folder.

Train.py :
  - Line 14: The epochs used for the training of this model is 3000, however it can be changed if the user dims it necessary.
  - Line 18: Please change the path to where you must have saved the training data in the folder.
  - Line 19: Please change the path to where you must have saved the test data in the folder.
  - Line 96: Please change the path to where you want the model saved if you intend to run the train.py file.

