from enum import Enum


LOCAL_FRONT_END_URL = "http://localhost:3000"
PRODUCTION_FRONT_END_URL = "https://clinicalbuddy.co.uk"


class UserRoles(Enum):
    ADMIN = "admin"
    USER = "user"


class UserClusters(Enum):
    ADMIN_CLUSTER = "admin_cluster"
    UHCW_CLUSTER = "uhcw_cluster"
    PERSONAL_CLUSTER = "personal_cluster"


MED_PROMPTS = "Anemia, Warfarin, Thrombosis, Lisinopril, Pneumonia, Metformin, Hypertension, Atorvastatin, Myocardial infarction, Amlodipine, Arrhythmia, Levothyroxine, Bronchitis, Sertraline, Gastroenteritis, Omeprazole, Dyspnea, Metoprolol, Hypoglycemia, Insulin, Leukemia, Prednisone, Arthritis, Gabapentin, Hepatitis, Ibuprofen, Encephalitis, Aspirin, Glaucoma, Albuterol, Asthma, Pantoprazole, Fibromyalgia, Fluoxetine, Nephritis, Citalopram, Pancreatitis, Tramadol, Osteoporosis, Zolpidem, Migraine, Candesartan, Parkinson's disease, Donepezil, Schizophrenia, Quetiapine, Insomnia, Trazodone, Epilepsy, Carbamazepine, Chronic obstructive pulmonary disease (COPD), Fluticasone, Diarrhea, Loperamide, Hyperthyroidism, Methimazole, Erectile dysfunction, Sildenafil, Urinary tract infection (UTI), Ciprofloxacin, Depression, Escitalopram, Anxiety, Buspirone, Irritable bowel syndrome (IBS), Dicyclomine, Rheumatoid arthritis, Methotrexate, Psoriasis, Methoxsalen, Hypothyroidism, Thyroxine, Gout, Colchicine, Congestive heart failure (CHF), Furosemide, Deep vein thrombosis (DVT), Rivaroxaban, Opioid dependence, Buprenorphine, Attention deficit hyperactivity disorder (ADHD), Methylphenidate, Alzheimer's disease, Memantine, Hypertensive heart disease, Hydralazine, Coronary artery disease, Clopidogrel, Ovarian cancer, Tamoxifen, Prostate cancer, Bicalutamide, Breast cancer, Anastrozole, Lung cancer, Erlotinib, Leukemia, Imatinib, Melanoma, Dabrafenib, shortness of breath"


DOMAINS = [
    "https://cks.nice.org.uk/",
    "https://www.nice.org.uk/guidance",
    "https://dermnetnz.org/",
    "https://www.pcds.org.uk/",
    "https://pubmed.ncbi.nlm.nih.gov/",
    "https://www.cochranelibrary.com/",
    "https://www.fsrh.org/standards-and-guidance/",
    "https://patient.info/patientplus",
    "https://www.pcsg.org.uk/",
    "https://gpnotebook.com/",
    "https://www.sign.ac.uk/our-guidelines/",
    "https://rightdecisions.scot.nhs.uk/scottish-palliative-care-guidelines/",
    "https://www.westmidspallcare.co.uk/",
    "https://rightdecisions.scot.nhs.uk/scottish-palliative-care-guidelines/",
    "https://gpifn.org.uk/",
    "https://www.brit-thoracic.org.uk/quality-improvement/guidelines/",
    "https://labtestsonline.org.uk/",
    "https://litfl.com/",
    "https://www.msdmanuals.com/en-gb/professional",
    "https://www.rightbreathe.com/",
    "https://www.nhs.uk/conditions/",
    "https://www.nhsinform.scot/illnesses-and-conditions",
    "https://www.nhsinform.scot/self-help-guides",
    "https://patient.info/",
]


# def generate_messages(response_prompt, question_history):
#     messages = [
#         {
#             "role": "system",
#             "content": " Utilize use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability. Never add reference.",
#         },
#         {"role": "user", "content": response_prompt},
#     ]

#     if question_history:
#         for user_question in question_history:
#             messages.append({"role": "user", "content": user_question})

#     return messages
