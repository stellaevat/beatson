# Beatson BioProject Annotation Web-Application

## Aim
The **Beatson BioProject Annotation Web-Application** aims to help researchers at the Beatson Institute for Cancer Research to find entries from the NCBI BioProject database that are relevant to their research, despite their inconsistent and/or incomplete metadata fields. It can be used to annotate and predict annotations for BioProjects, with fields that serve the researchers' specific research goals.

## Features
### Annotate
The **Annotate** tab allows users to explore a subset of the BioProject database, to annotate it with their fields of interest.

• There are search and filter functionalities to help discover relevant projects and view details about them.

• New labels can be created for annotation, or chosen from a drop-down list of the labels already used.
![annotate](https://github.com/stellaevat/beatson/assets/97710362/c14e52e3-ad15-4b7b-a73a-6f8806c8f818)

### Search
The **Search** tab allows to search for more projects, directly from the full BioProject database, to supplement the local dataset and strengthen annotations.

• The search results can be added to the dataset, so they can be annotated in the **Annotate** tab.

• All search results at once, or a smaller selection, can be added to the dataset.
![search](https://github.com/stellaevat/beatson/assets/97710362/8aad2675-67a0-4962-a5db-a9f660d6e88b)

### Predict
The **Predict** tab allows to run a prediction algorithm to get predictions for all unannotated projects, based on the existing annotations.

• Each prediction comes with a score for the algorithm's confidence in it.

• Predictions can be confirmed or corrected through manual annotation.

• To best improve the future performance of the algorithm, users are given suggestions for the projects to annotate next, based on lowest confidence score.
![predict](https://github.com/stellaevat/beatson/assets/97710362/5852e3ab-6edb-4101-a7c3-7db2ab845439)
 
## Set-up

- **Create three private Google sheets:** Log in to [Google Sheets](https://sheets.google.com), create 3 new blank sheets and ensure that general access is restricted. Create the following columns in each of them (or as otherwise outlined at the top of **features/gsheets.py**), and select **Plain text** format for all cells in all sheets.

```
Project Data:       UID, Accession, Title, Name, Description, Data_Type, Scope, Organism, PMIDs, Annotation, Prediction, Score, To_Annotate
Publication Data:   PMID, Title, Abstract, MeSH, Keywords
Prediction Metrics: Date, Dataset_Hash, Train_Size, Test_Size, F1_micro, F1_macro
```

- **Create a Google Cloud project:** Log in to [Google Cloud](https://cloud.google.com), go to the [Console](https://console.cloud.google.com/welcome) and create a new project.
- **Enable Google Sheets API:** Go to the [API Library](https://console.cloud.google.com/apis/library), find the [Google Sheets API Service](https://console.cloud.google.com/apis/library/sheets.googleapis.com) and with the new project selected to work on, enable the service.
- **Create a service account and key:** Go to [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts), select the project to work on, and create a service account, granting *Editor* permissions. Under *Actions* select *Manage keys*, create a new key in JSON format and save it in a secure location on your device.
- **Share the google sheets with the service account:** Copy the email address of the service account created and give it *Editor* permissions on the three Google sheets created earlier.
- **Create streamlit secrets:** Create a **secrets.toml** file in the hidden **.streamlit** directory found in the root directory of the app and paste the contents of the JSON service account key downloaded earlier. The contents should be converted to the template format shown below, with the addition of the google sheet URLs at the top.

```
private_gsheets_url_proj = "<PROJECT_DATA_SHAREABLE_LINK_WITH_EDIT_ACCESS>"
private_gsheets_url_pub = "<PUBLICATION_DATA_SHAREABLE_LINK_WITH_EDIT_ACCESS>"
private_gsheets_url_metrics = "<PREDICTION_METRICS_SHAREABLE_LINK_WITH_EDIT_ACCESS>"

[gcp_service_account]
type = "service_account"
project_id = "<KEY_DATA>"
private_key_id = "<KEY_DATA>"
private_key = "<KEY_DATA>"
client_email = "<KEY_DATA>"
client_id = "<KEY_DATA>"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "<KEY_DATA>"
universe_domain = "googleapis.com"
```
- **Deploy app:** Log in and deploy the app on [Streamlit.io](https://share.streamlit.io/), using the link to the current repository, the **main** branch and **main.py** as the main file. In the *Advanced Settings*, set the Python version to **3.11.3** and paste the contents of **.streamlit/secrets.toml** under *Secrets*.
- **Set up local repo:** Clone the current repository, navigate to the root directory and run **pip install -r requirements.txt** to install all dependencies (ideally in a dedicated virtual environment). To populate the Google sheets with the random project IDs listed in **project_ids.txt**, run **python populate.py**. Update the *Entrez.email* in line 10 of **features/search.py** with your own.
- **Track changes:** Run **streamlit run main.py** to run the app locally and observe the effects of your changes. Changes need to be pushed to the remote repository to be reflected in the deployed app.
