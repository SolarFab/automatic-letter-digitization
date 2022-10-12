# letter_automization

## context
The goal of this project is to develop a prototype for the digitization of postal letters. The app should make it possible to automatically recognize the sender via camera and a CNN model, scan the individual pages of the letter in the next step and then automatically save them in the correct folders.

## concept
- tensorflow CNN: using the mobilenetV2 neural network which is optimized for image and object detection
- three extra layers were added
- for the prototype logos of six companies were used to train the mobilenetV2 model
- python flask framework was used to build a webbased user interface 
- open source python pdf scanner was used to scan the documents and save it as pdf

### First Step: Take picture of logo o letter head


![Bildschirmfoto vom 2022-10-12 21-11-32](https://user-images.githubusercontent.com/101807190/195428240-3bbe5e57-654a-439a-971a-eeeb53bad372.png)


### Second Step: CNN predicts the sender 

![Bildschirmfoto vom 2022-10-12 21-08-08](https://user-images.githubusercontent.com/101807190/195428712-adfd6342-fc2d-4bf4-bebd-4cf236fb60a0.png)


### Third Step: Scan all the pages of the document

![Bildschirmfoto vom 2022-10-12 21-22-08](https://user-images.githubusercontent.com/101807190/195429932-9668d1e3-7e25-42a8-aa6a-95bc994fc417.png)

### Save: Document is automatically saved in respective company folder

![Bildschirmfoto vom 2022-10-12 21-26-39](https://user-images.githubusercontent.com/101807190/195430797-13e08832-c75f-42fc-97f1-7d567ddf0cd1.png)
