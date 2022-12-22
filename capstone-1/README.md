
# Content
1. Problem definition
2. How to run the service
3. Included files in this repository

## 1. Problem definition
Identify the risk of having a stroke based on a list of input information:
 + Age
 + Gender
 + Having heart disease?
 + Having hypertension?
 + Ever married?
 + BMI index (Body Mass index)
 + Residence type? (Rural/Urban)
 + Smoking status
 + Employment status (Work type)

## 2. How to run the project
There are two way to run the project

1. Deploy and run Docker image  
Get the Dockerfile in this repository and run below command, with whatever tag name. In the example tag name is v01
   ```docker build -t stroke-prediction:v01```
  Then run the docker image with below command
   ```docker run -it --rm -p 9696:9696 stroke-prediction:v01```
    
 
2. Run the API service   
The Machine learning model has been deployed into an API service using Google cloud.
An API test tool like Postman can be used to manipulate with this API.
At the time of submitting this midterm project for grading (Nov 08, 2022), the API service is still up.
However, the service will be suspended after the grading period because of insufficient maintenance cost.

+ URL: https://stroke-prediction-nihetndziq-uc.a.run.app/api/stroke-predict
+ Method: POST
+ Sample Payload  
{
 "Age": 80,<br/> 
 "Gender": "male" <male/female/not sure>, <br/> 
 "Heart_disease" : "yes" <yes/no>,<br/> 
 "Hypertension": "yes"  <yes/no>,<br/> 
 "Ever_married": "yes"  <yes/no>,<br/> 
 "Avg_glucose_level": 180,<br/> 
 "Bmi": 40,<br/> 
 "Urban_residence": "yes" <yes/no>,<br/> 
 "Smoking_status": "yes" <yes/no/formally smoked>,<br/> 
 "Work_type": "Self-employed"  <Private, Self-employed, Private, Govt_job><br/> 
  
}

![image](https://user-images.githubusercontent.com/58269366/200979743-db76f6bc-61dc-4134-9a58-8ed7aa815364.png)


## 3. Included files
+ Stroke-prediction-notebook.ipynb
+ Data/stroke-data.csv
+ Models/stroke_predict_rfc.pkl
+ app.py
+ Dockerfile
+ Pipfile
+ Pipfile.lock




