import tkinter as tk
from tkinter import simpledialog
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Global variables
env_file = ".env"
master_password_hash = "e7bc2f973afb8dfaf00fadfb19596741108be08ab4a107c6a799c429b684c64a"  

def hash_password(password):
    # Hash the password using SHA-256 algorithm
    return hashlib.sha256(password.encode()).hexdigest()

def get_last_password_change():
    global env_file
    try:
        last_modified_time = os.path.getmtime(env_file)
        return datetime.fromtimestamp(last_modified_time)
    except FileNotFoundError:
        # Return a default time if the .env file doesn't exist
        return datetime.now() - timedelta(days=365)

def authenticate():
    global last_password_change
    last_password_change = get_last_password_change()
    print("last_password_change",last_password_change)

    # Check if the password has expired (every 2 minutes in this example)
    if (datetime.now() - last_password_change) > timedelta(minutes=2):
        if change_password():
            last_password_change = datetime.now()

    else:
        # Create a Tkinter window
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # # Prompt the user for a password using a pop-up dialog
        password = simpledialog.askstring("Password", "Enter your password:", show='*')
        current_password=os.getenv("PASSWORD")

        print("password from UI",password)
        print("current_password from .env",current_password)
        
        # Check if the password is correct (you would replace this with your own validation logic)
        if password == current_password:
            simpledialog.messagebox.showinfo("Success", "Authentication successful!")
            return True
        else:
            simpledialog.messagebox.showerror("Error", "Incorrect password. Authentication failed.")
            return False

def change_password():
    # Prompt the user for the master password to change the expired password
    entered_master_password = simpledialog.askstring("Master Password", "Enter master password to change expired password:", show='*')
    entered_master_password_hash = hash_password(entered_master_password)
    print("entered_master_password_hash",entered_master_password_hash)
    print("master_password_hash",master_password_hash)

    if entered_master_password_hash == master_password_hash:
        new_password = simpledialog.askstring("New Password", "Enter your new password:", show='*')
        confirm_password = simpledialog.askstring("Confirm Password", "Confirm your new password:", show='*')

        if new_password == confirm_password:
            # Store the new password hash in .env file (you may replace this with your own storage method)
            with open('.env', 'w') as env_file:
                env_file.write(f"PASSWORD={new_password}")
            simpledialog.messagebox.showinfo("Success", "Password changed successfully!")
            return True
        else:
            simpledialog.messagebox.showerror("Error", "Passwords do not match. Password change failed.")
            return False
    else:
        simpledialog.messagebox.showerror("Error", "Incorrect master password. Password change failed.")
        return False

def main():
    if authenticate():
        print("Authentication successful. Proceeding with the script...")
        # Add your script logic here
    else:
        print("Authentication failed. Exiting...")

if __name__ == "__main__":
    main()
