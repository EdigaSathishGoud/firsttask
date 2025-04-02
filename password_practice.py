import tkinter as tk
from tkinter import simpledialog
from datetime import datetime, timedelta

# Global variable to store the last time the password was changed
last_password_change = datetime.now()

def authenticate():
    global last_password_change

    # Check if the password has expired (every 2 minutes in this example)
    if (datetime.now() - last_password_change) > timedelta(minutes=2):
        simpledialog.messagebox.showwarning("Password Expired", "Your password has expired. Please change it.")
        last_password_change = datetime.now()

    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt the user for a password using a pop-up dialog
    password = simpledialog.askstring("Password", "Enter your password:", show='*')

    # Check if the password is correct (you would replace this with your own validation logic)
    if password == "sa123":
        simpledialog.messagebox.showinfo("Success", "Authentication successful!")
        return True
    else:
        simpledialog.messagebox.showerror("Error", "Incorrect password. Authentication failed.")
        return False

def main():
    if authenticate():
        print("Authentication successful. Proceeding with the script...")
        # Add your script logic here
    else:
        print("Authentication failed. Exiting...")

if __name__ == "__main__":
    main()
