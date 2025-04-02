import psycopg2

#Connecting Database
conn = psycopg2.connect(host="localhost",dbname="Customer_360",user="postgres",password="SQL@469",port=5432)

#To execute commands
cur=conn.cursor()


# Create Table employees
table_info=""" 

/*TABLE 1: Customer*/

CREATE TABLE Customer
(
Customer_ID BIGINT NOT NULL,
Cust_Add_ID BIGINT NOT NULL,
First_Name VARCHAR(255) NOT NULL,
Last_Name VARCHAR(255) NOT NULL,
Gender VARCHAR(100),
Date_of_Birth  VARCHAR(20),
SSN VARCHAR(50),
Personal_Email VARCHAR(100),
Mobile VARCHAR(50),
Block_Number VARCHAR(50),
Personal_Addr_Line_1 VARCHAR(50),
Personal_Street VARCHAR(50),
Personal_ZIP_Code BIGINT,
Personal_City VARCHAR(50),
Business_Addr_Line_1 VARCHAR(50),
Business_Street VARCHAR(50),
Business_ZIP_Code BIGINT,
Business_City VARCHAR(50),
Marital_Status VARCHAR(50),
Education_Level VARCHAR(100),
Occupation VARCHAR(50),
Customer_Registration_Date DATE,
Email_Preferred VARCHAR(10),
Phone_Preferred VARCHAR(10),
Customer_Since DATE,
Loyalty_Program_Membership VARCHAR(50),
InstaHandle VARCHAR(100),
Facebook_Handle VARCHAR(100),
Twitter_Handle VARCHAR(100),
Hobby_1 VARCHAR(50),
Hobby_2 VARCHAR(50),
Areas_of_Interest VARCHAR(50),
Preferred_Payment_Methods VARCHAR(50),
LastInteractionDate VARCHAR(50),
Business_Email_Address VARCHAR(100),
PreferredEmailAddress VARCHAR(100),
AlternateEmailAddress VARCHAR(100),
MailingAddress VARCHAR(100),
PRIMARY KEY(Customer_ID)
);

/*TABLE 2: Leads*/

CREATE TABLE Leads
(
Lead_ID BIGINT NOT NULL,
Customer_ID BIGINT NOT NULL,
Source VARCHAR(50),
Status VARCHAR(50),
Owner VARCHAR(100),
Lead_Date VARCHAR(50),
Description VARCHAR(100),
Approx_Budget BIGINT,
Priority VARCHAR(50),
Next_Action_Date DATE,
Customer_Mobile VARCHAR(50),
PRIMARY KEY(Lead_ID),
FOREIGN KEY (Customer_ID) REFERENCES Customer(Customer_ID)
);
 
/*TABLE 3:Channel_Sale */

CREATE TABLE Channel_Sale
(
Channel_Sale_ID BIGINT,
Customer_ID BIGINT,
Partner_ID VARCHAR(10),
Sales_Rep_ID VARCHAR(10),
Channel_Partner_Name VARCHAR(100),
Sales_Channel VARCHAR(50),
Sale_Date DATE,
Sales_Revenue BIGINT,
Commission_Percentage INT,
Notes VARCHAR(100),
Channel_Performance_Metrics VARCHAR(100),
Region VARCHAR(50),
Channel_Partner_Relationship_Manager VARCHAR(50),
PRIMARY KEY(Channel_Sale_ID),
FOREIGN KEY (Customer_ID) REFERENCES Customer(Customer_ID)
);

/*TABLE 4: Opportunity*/

CREATE TABLE Opportunity
(
Opportunity_ID BIGINT NOT NULL,
Customer_ID BIGINT NOT NULL,
Lead_ID BIGINT NOT NULL,
Channel_Sale_ID BIGINT,
Customer_Mobile VARCHAR(50),
Channel VARCHAR(50),
Opportunity_Name VARCHAR(100),
Sales_Stage VARCHAR(100),
Close_Date DATE,
Opportunity_Owner VARCHAR(100),
Product VARCHAR(100),
PRIMARY KEY(Opportunity_ID),
FOREIGN KEY (Customer_ID) REFERENCES Customer(Customer_ID),
FOREIGN KEY (Lead_ID) REFERENCES Leads(Lead_ID),
FOREIGN KEY (Channel_Sale_ID) REFERENCES Channel_Sale(Channel_Sale_ID)

);


/*TABLE 5: Pricing Deals*/

CREATE TABLE Pricing_Deals
(
Deal_ID BIGINT NOT NULL,
Customer_ID BIGINT NOT NULL,
Opportunity_ID BIGINT NOT NULL,
Channel_Sale_ID BIGINT NOT NULL,
Product_Name VARCHAR(100),
Sales_Amount BIGINT,
Discount_Percentage BIGINT,
Notes VARCHAR(255),
Contract_Expiry_Date DATE,
Payment_Terms VARCHAR(100),
Renewal_Options VARCHAR(100),
PRIMARY KEY(Deal_ID),
FOREIGN KEY(Customer_ID) REFERENCES Customer(Customer_ID),
FOREIGN KEY (Channel_Sale_ID) REFERENCES Channel_Sale(Channel_Sale_ID),
FOREIGN KEY (Opportunity_ID) REFERENCES Opportunity(Opportunity_ID)

);

/*TABLE 6: Product*/

CREATE TABLE Product
(
Product_ID BIGINT NOT NULL,
Name VARCHAR(100),
Product_Category VARCHAR(50),
Product_Description VARCHAR(100),
Manufacturer VARCHAR(50),
Supplier VARCHAR(100),
Unit_Price VARCHAR(50),
PRIMARY KEY(Product_ID)
);

/*TABLE 7: Product_Details*/

CREATE TABLE Product_Details
(
Product_ID BIGINT NOT NULL,
Variation VARCHAR(100),
Stock BIGINT,
Colour VARCHAR(25),
PRIMARY KEY(Product_ID)
);

/*TABLE 8: Contract*/

CREATE TABLE Contract
(
Contract_ID BIGINT NOT NULL,
Customer_ID BIGINT NOT NULL,
Product_ID BIGINT NOT NULL,
Contract_Start_Date DATE,
Contract_End_Date DATE,
Contract_Term BIGINT,
Contract_Status VARCHAR(50),
Billing_Frequency VARCHAR(50),
Billing_Amount BIGINT,
Auto_Renewal VARCHAR(10),
Renewal_Notification_Days BIGINT,
Payment_Method_ID VARCHAR(10),
Contract_Notes VARCHAR(255),
PRIMARY KEY(Contract_ID),
FOREIGN KEY(Customer_ID) REFERENCES Customer(Customer_ID),
FOREIGN KEY(Product_ID) REFERENCES Product(Product_ID)

);


/*TABLE 9: Supplier*/

CREATE TABLE Supplier
(
Supplier_ID BIGINT,
Name VARCHAR(100),
Location_City VARCHAR(100),
Status VARCHAR(20),
Category VARCHAR(100),
Email VARCHAR(100),
Business_Phone VARCHAR(50),
Type VARCHAR(100),
Onboarded_Date DATE,
License_Number VARCHAR(50),
PRIMARY KEY(Supplier_ID)
);


/*TABLE 10: Invoice*/

CREATE TABLE Invoice
(
Invoice_ID BIGINT,
Customer_ID BIGINT,
Channel_Sale_ID BIGINT,
Product_ID BIGINT,
Partner_ID VARCHAR(50),
Supplier_ID BIGINT,
Invoice_Number VARCHAR(50),
Invoice_Date DATE,
Total_Amount BIGINT,
Amount_Paid BIGINT,
Payment_Status VARCHAR(50),
Payment_Method VARCHAR(50),
Payment_Date VARCHAR(50),
Notes VARCHAR(100),
Discounts BIGINT,
Tax_Amount BIGINT,
Currency VARCHAR(10),
Invoice_Status VARCHAR(20),
PRIMARY KEY(Invoice_ID),
FOREIGN KEY(Customer_ID) REFERENCES Customer(Customer_ID),
FOREIGN KEY (Channel_Sale_ID) REFERENCES Channel_Sale(Channel_Sale_ID),
FOREIGN KEY(Product_ID) REFERENCES Product(Product_ID),
FOREIGN KEY(Supplier_ID) REFERENCES Supplier(Supplier_ID)

);



/*TABLE 11: Shipment*/

CREATE TABLE Shipment
(
Shipment_ID BIGINT NOT NULL,
Invoice_ID BIGINT NOT NULL,
Shipment_Date DATE,
Expected_Arrival_Date DATE,
Actual_Arrival_Date DATE,
Shipment_Status VARCHAR(50),
Carrier VARCHAR(50),
Tracking_Number VARCHAR(50),
Shipping_Cost VARCHAR(50),
Cust_Add_ID BIGINT,
Shipping_Method VARCHAR(50),
Delivery_Notes VARCHAR(50),
PRIMARY KEY(Shipment_ID),
FOREIGN KEY(Invoice_ID) REFERENCES Invoice(Invoice_ID)
);

"""
cur.execute(table_info)


conn.commit()

cur.close()

conn.close()

