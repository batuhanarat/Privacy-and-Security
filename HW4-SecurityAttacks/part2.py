import hashlib
import pandas as pd
from part1 import load_rockyou

def hash_password(password, salt):
    combined =  salt + password  # Assuming salt is prepended
    return hashlib.sha512(combined.encode('utf-8')).hexdigest()

def load_salty_digitalcorp(file_path):
    df = pd.read_csv(file_path, names=['username', 'salt', 'hash_of_password'])
    return df

def dictionary_attack_for_salted(rockyou_path, salty_digitalcorp_path):
    rockyou_passwords = load_rockyou(rockyou_path)
    salty_digitalcorp_data = load_salty_digitalcorp(salty_digitalcorp_path)

    cracked_passwords = {}

    for index, row in salty_digitalcorp_data.iterrows():
        user_salt = row['salt']
        user_hash = row['hash_of_password']
        for password in rockyou_passwords:
            hashed = hash_password(password, user_salt)
            if hashed == user_hash:
                cracked_passwords[row['username']] = password
                break

    return cracked_passwords

if __name__ == "__main__":
    rockyou_path = 'rockyou.txt'
    salty_digitalcorp_path = 'salty-digitalcorp.txt'

    cracked_passwords = dictionary_attack_for_salted(rockyou_path, salty_digitalcorp_path)

    for username, password in cracked_passwords.items():
        print(f'Username: {username}, Password: {password}')