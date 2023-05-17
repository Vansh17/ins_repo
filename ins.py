
# # ************************CAESAR CIPHER*********************
def encrypt(message,key):
    encrypted_message=""
    for letter in message:
        if letter.isupper():
            encrypted_message+=chr((ord(letter) +key-64)%26+65)
        else:
            encrypted_message+=chr((ord(letter) +key-96)%26+97)
    return encrypted_message

plain_text="Hellmyname"
key=5
print("Plain text: ",plain_text)
print("Key: ",key)
print("\n")

def decrypt(cipher_text,key):
    decrypted_message=""
    for letter in cipher_text:
        if letter.isupper():
            decrypted_message+=chr((ord(letter) + key - 65) % 26 +65)
        else:
            decrypted_message+=chr((ord(letter) + key - 97) % 26+ 97)
    return decrypted_message

def brute_force(cipher_text):
    for i in range(26):
        print("Key is: ",abs(25-i))
        print("Decrypted text: "+decrypt(cipher_text,i))

brute_force(encrypt(plain_text,key))


            

# ***********************DIFFIE HELLMAN****************** 
prime=int(input("Enter a prime number: "))
g=int(input("Ente primitive root (g<p)"))
PkXa=int(input("Enter the private key of A: "))
PkXb=int(input("Enter the private key of B: "))
ya=g**PkXa %prime
yb=g**PkXb %prime
ka=yb**PkXa % prime
kb=ya**PkXb % prime
print("Public key of A: ",ya)
print("Public key of B: ",yb)
print("Shared secret key k: ",ka)
print("Shared secret key k: ",kb)


# ******************HILL CIPHER***********************
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

def modinv(a,m):
    g,x,y=egcd(a,m)
    if g!=1:
        raise Exception("Inverse does not exist")
    else:
        return x%m
    
def egcd(a,b):
    if a==0:
        return (b,0,1)
    else:
        g,y,x=egcd(b%a,a)
        return(g,x- (b//a)*y,y)
    

key= int(input("Enter the value of key: "))
modi=modinv(key,256)

images=asarray(Image.open(r'hello.png'))
images=images*key
plt.imshow(images)
plt.show()

images=images*modi
plt.imshow(images)
plt.show()

# ***********************RSA *******************************

import math
def gcd(a,h):
	temp=0
	while(1):
		temp=a%h
		if(temp==0):
			return h
		else:
			a=h
			h=temp
p=3
q=7
n=p*q
phi=(p-1)*(q-1)
print("n= ",n)
print("phi(n)= ",phi)
e=2
while(e<phi):
	if(gcd(e,phi)==1):
		break
	else:
		e+=1
k=2
d=(1+(k*phi))/e
print("Public key is ",{e,n})
print("Private key is ",{d,n})
msg=12
print("Plain text is ",msg)
ct=pow(msg,e)
ct=math.fmod(ct,n)
print("cipher text is ",ct)
pt=pow(ct,d)
pt=math.fmod(pt,n)
print("plain text is ",pt)

# *******************PLAYFAIR********************
m=[[0 for i in range(5)] for i in range(5)]
plain_text='HELLOWORLD'
key="BIRTHDAY"
plain_text=plain_text.replace("J","I")
key=key.replace("J","I")
d=[0 for i in range(26)]
row=0
column=0
for i in key:
    c=ord(i)-65
    if c>=74:
        c=c-1
    elif d[c]:
        continue
    else:
        if(column==5):
            row+=1
            column=0
        if(i=='J'):
            m[row][column]='I'
        else:
            m[row][column]=i
        column+=1
        d[c]=1
t=0
for i in d:
    if i or t==9:
        t+=1
        continue
    else:
        if(column==5):
            row+=1
            column=0
        m[row][column]=chr(t+65)
        column+=1
        d[c]=1
    t+=1


print("MATRIX : [")
for i in m:
    print(i)
print("]")
temp=[j for t in m for j in t]
if(len(plain_text)%2!=0):
    plain_text+='X'
s=[]
for i in range(0,len(plain_text)-1,2):
    s.append(plain_text[i]+plain_text[i+1])

m=''.join(j for i in m for j in i)

def encrypt():
    cipher=''
    pairs=s
    key_matrix=m
    for pair in pairs:
        row1,col1=divmod(key_matrix.index(pair[0]),5)
        row2,col2=divmod(key_matrix.index(pair[1]),5)
        if(row1==row2):
            cipher+=key_matrix[(row1*5)+(col1+1)%5]+key_matrix[(row2*5)+(col2+1)%5]
        elif (col1==col2):
            cipher+=key_matrix[((row1+1)%5)*5+col1]+key_matrix[((row2+1)%5)*5+col2]
        else:
            cipher+=key_matrix[(row1*5)+col2]+key_matrix[(row2*5)+col1]
    return cipher

print("cipher text is ",encrypt())



def decrypt(cipher):
    plain_text=''
    pairs=s
    key_matrix=m
    for pair in [cipher[i:i+2] for i in range(0,len(cipher),2)]:
        row1,col1=divmod(key_matrix.index(pair[0]),5)
        row2,col2=divmod(key_matrix.index(pair[1]),5)
        if(row1==row2):
            plain_text+=key_matrix[(row1*5)+(col1-1)%5]+key_matrix[(row2*5)+(col2-1)%5]
        elif (col1==col2):
            plain_text+=key_matrix[((row1-1)%5)*5+col1]+key_matrix[((row2-1)%5)*5+col2]
        else:
            plain_text+=key_matrix[(row1*5)+col2]+key_matrix[(row2*5)+col1]
    return plain_text

print("plain text is ",decrypt(encrypt()))

# ********************COLUMNAR********************************


import math
key = "birthday"

def decrypt(cipher_text):
    decry_msg = ""
    key_ind = 0
    msg_indx = 0
    msg_num = float(len(cipher_text))
    msg_lst = list(cipher_text)
    col = len(key)
    row = int(math.ceil(msg_num / col))
    key_lst = sorted(list(key))
    deci_msg = []
    for _ in range(row):
        deci_msg += [[None] * col]
    for _ in range(col):    
        curr_idx = key.index(key_lst[key_ind])
        for j in range(row):
            deci_msg[j][curr_idx] = msg_lst[msg_indx]
            msg_indx += 1
        key_ind += 1
    try:
        decry_msg = ''.join(sum(deci_msg, []))
    except TypeError:
        raise TypeError("This program cannot","handle repeating words.")
    
    null_count = decry_msg.count('_')
    if null_count > 0:
        return decry_msg[: -null_count]

    return decry_msg



def encrypt(msg):
    cipher_text = ""
    key_ind = 0
    msg_num = float(len(msg))
    msg_lst = list(msg)
    key_lst = sorted(list(key))
    col = len(key)
    row = int(math.ceil(msg_num / col))
    fill_null = int((row * col) - msg_num)
    msg_lst.extend('_' * fill_null)
    matrix = [msg_lst[i: i + col] for i in range(0, len(msg_lst), col)]
    for _ in range(col):
        curr_idx = key.index(key_lst[key_ind])
        cipher_text += ''.join([row[curr_idx] for row in matrix])
        key_ind += 1
    return cipher_text




msg = "Vansh Dodiya"
print("\nPlaintext Message:", msg)
cipher_text = encrypt(msg)
print("\nCiphertext Message: {}".format(cipher_text))
decry_msg = decrypt(cipher_text)
print("\nDecrypted Message: {}\n".format(decry_msg))


# *********************ECB***************************
# Ecb INS
para = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print("The text is: ",para)
print("\n")
def decimalToBinary(n):
  ans = "{0:b}".format(int(n))
  while len(ans) <8:
    ans="0"+ans
  return ans

def binaryToDecimal(n):
    return int(n,2)

t = ""
for i in para:
  t+= str(decimalToBinary(ord(i)))

s = ""
c = 0

while c*128 < len(t):
  c+=1

raw_t = t
padded = len(t)

if c*128 > len(t):
  t += '1' + '0'*(c*128-len(t)-1) 

ans = []

for i in range(c):
  g = ""
  for j in range(128):
        g+= str(t[128*i +j])
  ans.append(g)

x = 2

f_ans = []
for i in ans:
  temp = i
  for j in range(x):
      temp = temp[-1] +temp[:len(temp)-1] 
  f_ans.append(temp)

enc = ""
for i in f_ans:
  for j in range(0,128,8):
    temp = i[j:j+8]
    enc += chr(binaryToDecimal(temp))

t  = ""
for i in enc:
  t+= str(decimalToBinary(ord(i)))

c = len(t) // 128
d_ans = []
for i in range(c):
  g = ""
  for j in range(128):
      g+=str(t[i*128+j])
  d_ans.append(g)
d_f_ans = []
for i in d_ans:
  temp = i
  for j in range(x):
      temp = temp[1:] + temp[0] 
  d_f_ans.append(temp)
s = ""
for i in d_f_ans:
  s+=i
dec = ""
s = s[:padded]
for i in range(0 , len(s) , 8):
  temp = s[i : i+8]
  dec += chr(binaryToDecimal(temp))
print("Encrypted text is: ",enc )
print("\n")
print("Decrypted text is: ", dec )


# *************************MERKEL********************
import hashlib

def merkel_root(tx_hashes):
    if len(tx_hashes) == 1:
        return tx_hashes[0]
    new_tx_hashes = []
    for i in range(0, len(tx_hashes), 2):
        tx_hash1 = tx_hashes[i]
        if i+1 < len(tx_hashes):
            tx_hash2 = tx_hashes[i+1]
        else:
            tx_hash2 = tx_hash1
        combined_hash = hashlib.sha256(hashlib.sha256(tx_hash1.encode('utf-8') + tx_hash2.encode('utf-8')).digest()).hexdigest()
        new_tx_hashes.append(combined_hash)
        print(new_tx_hashes)
        print("\n")
    return merkel_root(new_tx_hashes)


tx_hashes = ["1234", "5678", "90ab", "cdef"]
merkel_root_hash = merkel_root(tx_hashes)
print(merkel_root_hash)


# ****************OWN***********************
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rubiks_cube_key(block_size,n):
    """
    Generates a Rubik's cube pattern as a key
    for encrypting and decrypting an image.
    """
    key = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    for i in range(block_size):
        for j in range(block_size):
            if i < block_size // n and j < block_size // n:
                # key[i, j, 0] = 255  # yellow
                key[i, j, :] = 255  # blue
            elif i >= 2 * block_size // n and j < block_size // n:
                key[i, j, 1] = 255  # green
            elif i >= block_size // n and i < 2 * block_size // n and j < block_size // n:
                key[i, j, 2] = 255  # red
            elif i < block_size // n and j >= 2 * block_size // n:
                # key[i, j, :] = 255  # blue
                key[i, j, 0] = 255  # yellow
            elif i >= 2 * block_size // n and j >= 2 * block_size // n:
                key[i, j, 0] = 255  # orange
                key[i, j, 1] = 165
            elif i >= block_size // n and i < 2 * block_size // n and j >= 2 * block_size // n:
                key[i, j, :] = 255  # white
            else:
                key[i, j, :] = 255  # white
    return key



def encrypt_image(img_path, key):
    img = Image.open(img_path)
    width, height = img.size
    block_size = int(width / 6) # divide image into 6 blocks
    img_arr = np.array(img)

    # divide each block into 9 sub-blocks and encrypt them
    for i in range(6):
        for j in range(36):
            x_start = i * block_size + int(j / 3) * int(block_size / 3)
            y_start = int(j % 3) * int(height / 3)
            sub_block = img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3)]
            sub_block_key = np.resize(key, sub_block.shape) # resize key array to match sub-block shape
            sub_block = np.bitwise_xor(sub_block, sub_block_key) # simple XOR encryption
            img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3)] = sub_block
            print(f"Block ({i+1},{j+1}) encrypted.")

    # # divide each block into 27 sub-blocks and encrypt them
    # for i in range(6):
    #     for j in range(27):
    #         x_start = i * block_size + int(j / 9) * int(block_size / 3)
    #         y_start = int(j / 3) % 3 * int(height / 3)
    #         z_start = j % 3 * int(block_size / 3)
    #         sub_block = img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3), z_start:z_start+int(block_size/3)]
    #         sub_block_key = np.resize(key, sub_block.shape) # resize key array to match sub-block shape
    #         sub_block = np.bitwise_xor(sub_block, sub_block_key) # simple XOR encryption
    #         img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3), z_start:z_start+int(block_size/3)] = sub_block
    #         print(f"Block ({i+1},{j+1}) encrypted.")


    # save the encrypted image
    encrypted_img = Image.fromarray(img_arr)
    encrypted_img.save("encrypted_image2.png")
    print("Image encrypted successfully.")

def decrypt_image(img_path, key):
    img = Image.open(img_path)
    width, height = img.size
    block_size = int(width / 6) # divide image into 6 blocks
    img_arr = np.array(img)

    # divide each block into 9 sub-blocks and decrypt them
    for i in range(6):
        for j in range(36):
            x_start = i * block_size + int(j / 3) * int(block_size / 3)
            y_start = int(j % 3) * int(height / 3)
            sub_block = img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3)]
            sub_block_key = np.resize(key, sub_block.shape) # resize key array to match sub-block shape
            sub_block = np.bitwise_xor(sub_block, sub_block_key) # simple XOR decryption
            img_arr[x_start:x_start+int(block_size/3), y_start:y_start+int(height/3)] = sub_block
            print(f"Block ({i+1},{j+1}) decrypted.")

    # save the decrypted image
    decrypted_img = Image.fromarray(img_arr)
    decrypted_img.save("decrypted_image2.png")
    print("Image decrypted successfully.")

n=int(input("Enter n: "))
block_size = 216 # 6 blocks x 9 sub-blocks x 3 pixels per sub-block
key = rubiks_cube_key(block_size,n)
print(key)
key_img = Image.fromarray(key)
key_img.save("key2.png")
print("Image key .")
img_path = "/content/humans.png"
encrypt_image(img_path, key)
decrypt_image("encrypted_image2.png", key)
