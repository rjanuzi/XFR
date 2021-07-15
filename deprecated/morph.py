import dnnlib, pickle
import dnnlib.tflib as tflib
import os
import glob
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt


tflib.init_tf()
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)

model_dir = 'cache/'
model_path = [model_dir + f for f in os.listdir(model_dir) if 'stylegan-ffhq' in f][0]
print("Loading StyleGAN model from %s..." % model_path)

with dnnlib.util.open_url(model_path) as f:
    generator_network, discriminator_network, averaged_generator_network = pickle.load(f)

print("StyleGAN loaded & ready for sampling!")


def generate_images(generator, latent_vector, z=True):
    batch_size = latent_vector.shape[0]


    if z:  # Start from z: run the full generator network
        return generator.run(latent_vector.reshape((batch_size, 512)), None, randomize_noise=False, **synthesis_kwargs)
    else:  # Start from w: skip the mapping network
        return generator.components.synthesis.run(latent_vector.reshape((batch_size, 18, 512)), randomize_noise=False,
                                                  **synthesis_kwargs)

def plot_imgs(path,model, rows, columns,w):
  a=0
  alpha=0
  step=1.0/len(w)
  print(len(w))
  for i in range(rows):
    f, axarr = plt.subplots(1,columns, figsize = (20,8))
    for j in range(columns):
      img = generate_images(model, w[a],z = False)[0]
      a+=1
      axarr[j].imshow(img)
      axarr[j].axis('off')
      axarr[j].set_title('Alpha: %f' %float(alpha))

      img=Image.fromarray(img)
      auxiliar = str(alpha)
      auxiliar = '%.2f'%(alpha)
      img.save(path+str(auxiliar)+'.png')
      alpha += step
    #plt.show()

def morph(path,W1, W2, model, num_images):
    #vetora = np.load(W1).reshape(1, 18, -1)
    #vetorb = np.load(W2).reshape(1, 18, -1)
    vetora = W1.reshape(1, 18, -1)
    vetorb = W2.reshape(1, 18, -1)
    n = 1.0 / num_images
    alpha = 0
    l = []
    for i in range(num_images):
        w = alpha * vetora + (1 - alpha) * vetorb
        # img=generate_images(averaged_generator_network, w, z = False)[0]
        l.append(w)
        alpha += n

    plot_imgs(path,averaged_generator_network, int(num_images / 5), 5, l)
    return

input_dir = 'fei_latent_representations2\\'

pair_dir = 'fei_pair2\\'

images_dir = 'fei_generated_images2\\'

vectors = glob.glob(input_dir+'*.npy')

vectorsb = glob.glob(pair_dir+'*.npy')

images_list = glob.glob(images_dir+'*.jpg')

pair_images_list = glob.glob(pair_dir+'*.jpg')

num_images = 20

for i in range(len(vectors)):
    aux = vectors[i]
    vec = np.load(aux)
    name = aux.split(sep=input_dir)
    name = name[1]
    name2 = name.split(sep='.npy')
    name2 = name2[0]
    name = name.split(sep='-')
    name = name[0]
    #print(name)
    lista = [x for x in vectorsb if name in x]
    path = pair_dir + name2
    if not os.path.exists(path):
        os.mkdir(path)
    for j in range(len(lista)):
        aux2 = lista[j]
        vec2 = np.load(aux2)
        path2 = path + '\\input_'
        name3 = aux2.split(sep=pair_dir)
        name3 = name3[1]
        name3 = name3.split(sep='.jpg')
        name3 = name3[0]
        if('01' in name3) and not('02' in name3):
            path4 = path2 + '01b\\'
            path2 = path2 + '01a\\'
            path3 = path +'\\output_01\\'
            number1 = True
            number2 = False

        elif('02' in name3):
            path4 = path2 + '02b\\'
            path2 = path2 + '02a\\'
            path3 = path + '\\output_02\\'
            number2 = True
            number1 = False

        if not os.path.exists(path2):
            os.mkdir(path2)

        if not os.path.exists(path3):
            os.mkdir(path3)

        if not os.path.exists(path4):
            os.mkdir(path4)

        alpha = 0
        img0 = averaged_generator_network.components.synthesis.run(vec.reshape((1, 18, 512)), randomize_noise=False,
                                                  **synthesis_kwargs)
        img1 = averaged_generator_network.components.synthesis.run(vec2.reshape((1, 18, 512)), randomize_noise=False,
                                                  **synthesis_kwargs)
        img0 = Image.fromarray(img0[0])
        img1 = Image.fromarray(img1[0])
        n = 1.0/num_images
        for k in range(num_images):
            auxiliar = str(alpha)
            auxiliar = '%.2f'%(alpha)
            img0.save(path2 + str(auxiliar) + '.jpg')
            img1.save(path4 + str(auxiliar) + '.jpg')
            alpha += n
        morph(path3,vec,vec2,averaged_generator_network,num_images)

        '''
        if(j == 0 ):
            path2 = path + '\\output\\'
            if not os.path.exists(path2):
                os.mkdir(path2)
            morph(path2,vec,vec2,averaged_generator_network,num_images)

        elif(j == 1):
            path2 = path + '\\output2\\'
            if not os.path.exists(path2):
                os.mkdir(path2)
            morph(path2, vec, vec2, averaged_generator_network, num_images)
            
        alpha = 0
        n = 1.0 / num_images
        if(j == 0):
            path3 = path + '\\input\\'
            input_img = [x for x in ]
        elif(j == 1):
            path3 = path + '\\input2\\'
            
        if not os.path.exists(path3):
            os.mkdir(path3)
        for k in range(len(num_images)):
            img = 
            alpha += n
    '''

