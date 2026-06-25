# Logistic Regression Base Model compared with adding each components to the base model

<img width="1373" height="312" alt="image" src="https://github.com/user-attachments/assets/d3eb3157-f5f4-471f-973f-abdbdc37d7ec" />

# Random Forest 
<img width="1286" height="37" alt="image" src="https://github.com/user-attachments/assets/defa73d8-dfe2-491a-a994-4a8507fdcd58" />
<img width="1387" height="372" alt="image" src="https://github.com/user-attachments/assets/8de9f514-3167-4daf-8763-c282f89a551e" />
<img width="910" height="702" alt="image" src="https://github.com/user-attachments/assets/92cdb7e6-e6fa-4b40-888f-d9dfc15ce4b4" />

#XGBoost 
<img width="1252" height="37" alt="image" src="https://github.com/user-attachments/assets/fd4761fa-1752-4cab-8e57-5da4d609d8b0" />
<img width="1382" height="370" alt="image" src="https://github.com/user-attachments/assets/40f8de50-cacd-43d3-8ade-638b0f633c12" />
<img width="907" height="698" alt="image" src="https://github.com/user-attachments/assets/47955dd0-295b-49f7-82e7-087e1245ddc0" />



# Conclusion 
Evaluated the model with ROC - AUC, Hosmer–Lemeshow (H-L) test, net reclassification improvement (NRI) index and the integrated discrimination improvement (IDI) index. The AUC increased from 0.68 to 0.89, while adding the KL degree(ordinal) into the group. we concluded that KL degree(ordinal) is a most significant feature contributing to future knee replacement. 
Fit the Random Forest and XGBoost models with the OAI dataset. Comparing the metrics( AUC, Accuracy, precision, recall, specificity, F1 score, and Brier score), Random Forest is indicated as a better model for the OAI dataset to predict the future knee replacement. 
Both Random Forest and XGBoost models indicate that KL degree and esKOA are significant features to predict the future knee replacement. 






