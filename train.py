import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import requests
from PIL import Image
import io
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import threading
from queue import Queue
import time

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

#global queue
processed_items = []
items_to_process = Queue()
has_data_to_process = True

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    worker_thread_count = 1
    retry_for_failed = 2
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
   #     transforms.RandomCrop(args.crop_size),
   #     transforms.RandomHorizontalFlip(), 
        transforms.Scale(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.L1Loss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            processed_items = []
            threads = []
            has_data_to_process = True
            def do_request(item):
                position = item['position']
                #print(position)
                #print(item)
                retry = retry_for_failed
                while retry:
                    r = requests.post('http://localhost:4567/', data=item)
                    if r.status_code == 200:
                        pil = Image.open(io.BytesIO(r.content)).convert('RGB')
                        processed_items[position] = transform(pil)
                        #print(position, processed_items[position])
                        break
                    else:
                        print("shouldb be here")
                        time.sleep(2)
                        retry -= 1 
            # Set mini-batch dataset
            image_tensors = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            #print(images.size())
            #print(torch.equal(images[0] ,images[1]))
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(image_tensors)
            outputs = decoder(features, captions, lengths)
            codes = []
            def worker():
                while items_to_process.qsize() > 0 or has_data_to_process:
                    item = items_to_process.get()
                    if item is None:
                        break
                    do_request(item)
                    items_to_process.task_done()
                print("ended thread processing")
            for j in range(worker_thread_count):
                 t = threading.Thread(target=worker)
                 t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
                 t.start()
                 threads.append(t)
            for ii, image in enumerate(images):
                image_tensor = to_var(image.unsqueeze(0), volatile=True)
                feature = encoder(image_tensor)
                sampled_ids = decoder.sample(feature)
                sampled_ids = sampled_ids.cpu().data.numpy()
                sampled_caption = []
                for word_id in sampled_ids:
                    word = vocab.idx2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break
                sentence = ' '.join(sampled_caption)
                payload = {'code': sentence}
                data = {'position': ii, 'code': sentence}
                items_to_process.put(data)
                processed_items.append('failed')
                codes.append(sentence)
            has_data_to_process = False
            print(codes)
            print(items_to_process.qsize())
            print(image.size())
            print("waiting for threads")
            for t in threads:
                t.join()
            print("done reassembling images")
            for t in threads:
                t.shutdown = True
                t.join()
            bad_value = False
            for pi  in processed_items:
                if isinstance(pi, str) and pi == "failed":
                    bad_value = True
            if bad_value == True:
                print("failed conversion,skipping batch")
                continue
            output_tensor =  torch.FloatTensor(len(processed_items),3,images.size()[2],images.size()[3])
            for ii,image_tensor in enumerate(processed_items):
                output_tensor[ii] = processed_items[ii]
            output_var = to_var(output_tensor,False)
            target_var = to_var(images,False)
            #loss = criterion(output_var,target_var)
            print("loss")
            print(loss)

            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
