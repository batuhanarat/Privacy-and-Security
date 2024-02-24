import hashlib
import pandas as pd
from part1 import load_rockyou

def hash_password(combined):
    return hashlib.sha512(combined.encode('utf-8')).hexdigest()

def load_keystretching(file_path):
    return pd.read_csv(file_path, names=['username', 'salt', 'hash_outcome'])

def precompute_hashes(passwords, salts, max_iterations):
    precomputed_hashes = {}
    for password in passwords:
        for salt in salts:
            xi = ''
            key = (password, salt)
            precomputed_hashes[key] = [xi]
            for i in range(max_iterations):
                combined = salt + xi + password
                xi = hash_password(combined)
                precomputed_hashes[key].append(xi)
    return precomputed_hashes

def dictionary_attack(rockyou_path, keystretching_path):
    rockyou_passwords = load_rockyou(rockyou_path)
    keystretching_data = load_keystretching(keystretching_path)
    salts = keystretching_data['salt'].unique()
    max_iterations = 2000
    precomputed_hashes = precompute_hashes(rockyou_passwords, salts, max_iterations)

    cracked_passwords = {}
    for index, row in keystretching_data.iterrows():
        user_salt = row['salt']
        user_hash = row['hash_outcome']
        for password in rockyou_passwords:
            key = (password, user_salt)
            if key in precomputed_hashes:
                for iteration in range(1, max_iterations + 1):
                    possible_hash = precomputed_hashes[key][iteration]
                    if possible_hash == user_hash:
                        cracked_passwords[row['username']] = (password, iteration)
                        break
                if row['username'] in cracked_passwords:
                    break

    return cracked_passwords

if __name__ == "__main__":
    rockyou_path = 'rockyou.txt'
    keystretching_path = 'keystreching-digitalcorp.txt'

    cracked_passwords = dictionary_attack(rockyou_path, keystretching_path)

    for username, (password, iteration) in cracked_passwords.items():
        print(f'Username: {username}, Password: {password}, Iteration: {iteration}')
