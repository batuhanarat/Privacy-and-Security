import hashlib
import pandas as pd

def hash_password(password):
    return hashlib.sha512(password.encode('utf-8')).hexdigest()

def load_rockyou(file_path):
    df = pd.read_csv(file_path, header=None, names=['password'], encoding='latin-1')
    common_passwords = df['password'].tolist()
    return common_passwords


def load_digitalcorp(file_path):
    df = pd.read_csv(file_path, sep=',')
    return df


def dictionary_attack(rockyou_path, digitalcorp_path):
    rockyou_passwords = load_rockyou(rockyou_path)
    hashed_passwords = {hash_password(password): password for password in rockyou_passwords}
    digitalcorp_data = load_digitalcorp(digitalcorp_path)
    digitalcorp_data['cracked_password'] = digitalcorp_data['hash_of_password'].map(hashed_passwords)
    cracked_data = digitalcorp_data.dropna(subset=['cracked_password'])
    return cracked_data

if __name__ == "__main__":
    rockyou_path = 'rockyou.txt'
    digitalcorp_path = 'digitalcorp.txt'
    cracked_passwords = dictionary_attack(rockyou_path,digitalcorp_path)
    print(cracked_passwords[['username', 'cracked_password']])
