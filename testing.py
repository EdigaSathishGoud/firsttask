from faker import Faker
import csv
from datetime import datetime, timedelta

fake = Faker('en_US')

# Function to generate a realistic US phone number
def generate_phone_number():
    return f"+1-{fake.msisdn()[:10]}"

# Function to generate a date within a specific range
def generate_date(start_date, end_date):
    return fake.date_between(start_date=start_date, end_date=end_date).strftime('%Y-%m-%d')

# Predefined values for Education Level and Occupation
education_levels = ["Bachelor's Degree", "Master's Degree", "PhD", "High School Diploma"]
occupations = ["Doctor", "Teacher", "Engineer", "Lawyer", "Nurse"]

# Generate data for 1 row
data = []
customer_id = 1
cust_add_id = fake.random_int(min=1000, max=9999)
first_name = fake.first_name()
last_name = fake.last_name()
gender = fake.random_element(elements=["Male", "Female"])
dob = generate_date(start_date='-70y', end_date='-18y')
ssn = fake.ssn()
personal_email = f"{first_name.lower()}.{last_name.lower()}@example.com"
mobile = generate_phone_number()
block_number = fake.random_int(min=1, max=999)
personal_addr_line_1 = fake.secondary_address()
personal_street = fake.street_name()
personal_zip_code = fake.zipcode()
personal_city = fake.city()
business_addr_line_1 = fake.secondary_address()
business_street = fake.street_name()
business_zip_code = fake.zipcode()
business_city = fake.city()
marital_status = fake.random_element(elements=["Married", "Single", "Divorced", "Widowed"])
education_level = fake.random_element(elements=education_levels)
occupation = fake.random_element(elements=occupations)
customer_registration_date = generate_date(start_date='-5y', end_date='today')
email_preferred = fake.random_element(elements=["Y", "N"])
phone_preferred = fake.random_element(elements=["Y", "N"])
customer_since = generate_date(start_date=customer_registration_date, end_date='today')
loyalty_program_membership = fake.random_element(elements=["Bronze", "Silver", "Gold", "Platinum"])
insta_handle = f"{first_name.lower()}.{last_name.lower()}"
facebook_handle = f"{first_name.lower()}.{last_name.lower()}"
twitter_handle = f"{first_name.lower()}.{last_name.lower()}"
hobby_1 = fake.word()
hobby_2 = fake.word()
areas_of_interest = fake.word()
preferred_payment_methods = fake.random_element(elements=["PayPal", "Amazon Pay", "Credit Card", "Debit Card"])
last_interaction_date = generate_date(start_date=customer_since, end_date='today')
business_email_address = fake.email()
preferred_email_address = personal_email
alternate_email_address = fake.email()
mailing_address = f"{fake.building_number()} {fake.street_name()}, {fake.city()}, {fake.state_abbr()} {fake.zipcode()}"

data.append({
    "Customer ID": customer_id,
    "Cust Add ID": cust_add_id,
    "First Name": first_name,
    "Last Name": last_name,
    "Gender": gender,
    "Date of Birth": dob,
    "SSN": ssn,
    "Personal Email": personal_email,
    "Mobile": mobile,
    "Block Number": block_number,
    "Personal Addr Line 1": personal_addr_line_1,
    "Personal Street": personal_street,
    "Personal ZIP Code": personal_zip_code,
    "Personal City": personal_city,
    "Business Addr Line 1": business_addr_line_1,
    "Business Street": business_street,
    "Business ZIP Code": business_zip_code,
    "Business City": business_city,
    "Marital Status": marital_status,
    "Education Level": education_level,
    "Occupation": occupation,
    "Customer Registration Date": customer_registration_date,
    "Email Preferred (Y/N)": email_preferred,
    "Phone Preferred (Y/N)": phone_preferred,
    "Customer Since (Date)": customer_since,
    "Loyalty Program Membership": loyalty_program_membership,
    "InstaHandle": insta_handle,
    "Facebook Handle": facebook_handle,
    "Twitter Handle": twitter_handle,
    "Hobby 1": hobby_1,
    "Hobby 2": hobby_2,
    "Areas of Interest": areas_of_interest,
    "Preferred Payment Methods": preferred_payment_methods,
    "LastInteractionDate": last_interaction_date,
    "Business Email Address": business_email_address,
    "PreferredEmailAddress": preferred_email_address,
    "AlternateEmailAddress": alternate_email_address,
    "MailingAddress": mailing_address
})

# Write data to CSV
with open("generated_data.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)