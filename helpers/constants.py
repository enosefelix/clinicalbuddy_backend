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
