import torch
import torch.nn.utils as torch_utils



from copy import deepcopy
import numpy as np

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar


from ignite.engine import Events

from simple_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from simple_ntc.trainer import Trainer, MyEngine


from transformers import AutoTokenizer
from transformers import BertForSequenceClassification



class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config): # model, crit, optimizer, scheduler, self.config
#         self.model = model
#         self.crit = crit
#         self.optimizer = optimizer
#         self.config = config
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config) # 
        
        self.best_loss = np.inf
        self.best_model = None
        
        self.best_acc = 0
        self.best_model2 = None        
        
    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()

        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device)

        x = x[:, :engine.config.max_length]

        # Take feed-forward
        y_hat = engine.model(x, attention_mask=mask)[0]

        loss = engine.crit(y_hat, y)
        loss.backward()

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            # Take feed-forward
            y_hat = engine.model(x, attention_mask=mask)[0] #loss, logits = outputs[:2]

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }
    
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        # ,alpha=0.99666
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            ) # 내가 만든 metric이 결과로 나왔고 그것에 대한 RunningAverage 를 구하겠다는 말. 

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)
            
            

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))
                with open(engine.config.txt_fn, "a") as f:
                    f.write('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}\n'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))
                    

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Test - loss={:.4e} accuracy={:.4f} best_loss={:.4e} best_acc={:.4f}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                    engine.best_acc,
                ))
                with open(engine.config.txt_fn, "a") as f:
                    f.write('Test - loss={:.4e} accuracy={:.4f} best_loss={:.4e} best_acc={:.4f}\n'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                    engine.best_acc,
                ))                

    @staticmethod
    def attach_test(validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
                   
        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Test - loss={:.4e} accuracy={:.4f} '.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy']
                ))
 

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.

    @staticmethod
    def check_best_acc(engine):
        acc = float(engine.state.metrics['accuracy'])
        if acc >= engine.best_acc: # If current epoch returns upper validation acc,
            engine.best_acc = acc  # Update lowest validation loss.
            engine.best_model2 = deepcopy(engine.model.state_dict()) # Update best model weights.
          
               
    @staticmethod
    def save_model(engine, train_engine, config, index_to_label, **kwargs):
        name = config.model_fn.split('.')[0]
        torch.save(
            {
                'bert_bm_loss': engine.best_model,  
                'bert_bm_acc': engine.best_model2,
                'bert' : engine.model,
                'config': config,
                'classes': index_to_label,
                **kwargs
            }, f'{name}_epoch_{engine.state.epoch}.pth' 
        ) # engine.state.epoch  # f'{config.model_fn.split('.')[0]}_epoch_{engine.state.epoch}.pth'  # config.model_fn

    
class Epoch_end():

    def __init__(self, config):
        self.config = config

            
#     def get_acc(self,logits, y):
#         _, predicted = torch.max(logits.data, 1)
#         total = y.size(0)
#         correct = (predicted == y).sum().item()
        
#         return correct, total    
    

#     def read_text(self,fn):
#         with open(fn, 'r') as f:
#             lines = f.readlines()

#             labels, texts = [], []
#             for idx, line in enumerate(lines):
#                 if idx == 0:
#     #                 print(line) # header 지우기
#                     continue
#                 if line.strip() != '':
#                     # The file should have tab delimited two columns.
#                     # First column indicates label field,
#                     # and second column indicates text field.
#                     label, text = line.strip().split('\t')
#                     labels += [label]
#                     texts += [text]

#         return labels, texts
    
    @staticmethod
    def test(self, model, config):
        name = config.model_fn.split('.')[0]
        saved_data = torch.load(
            f'{name}_epoch_1.pth',
            map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
        )
        train_config = saved_data['config']
        bert = saved_data['bert']  # 'bert' 'bert_bm_loss' 'bert_bm_acc'
        index_to_label = saved_data['classes']

        label_to_index = {}
        for key, val in index_to_label.items():
            label_to_index[val] = key
        
        def read_text(fn):
            with open(fn, 'r') as f:
                lines = f.readlines()

                labels, texts = [], []
                for idx, line in enumerate(lines):
                    if idx == 0:
        #                 print(line) # header 지우기
                        continue
                    if line.strip() != '':
                        # The file should have tab delimited two columns.
                        # First column indicates label field,
                        # and second column indicates text field.
                        label, text = line.strip().split('\t')
                        labels += [label]
                        texts += [text]

            return labels, texts
        
        def get_acc(logits, y):
            _, predicted = torch.max(logits.data, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()

            return correct, total  
        
        # 테스트 데이터 
        labels, lines = read_text(config.test_fn) # list of texts

        # 정답 
        labels = list(map(label_to_index.get, labels))


        # test.tsv : labels , lines  : raw list 인풋
        with torch.no_grad():
            if config.gpu_id >= 0:
    #             model.cuda(config.gpu_id)
    #             device = torch.device("cuda")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         device = next(model.parameters()).device

#             # Declare model and load pre-trained weights.
            tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
#             model = BertForSequenceClassification.from_pretrained(
#                 train_config.pretrained_model_name,
#                 num_labels=len(index_to_label)
#             )
#             #매 에포트 학습된 모델로 갈아끼워넣기
#             model.load_state_dict(bert)
            model.to(device)

            # Don't forget turn-on evaluation mode.
            model.eval()      

            y_hats = []

            tot_correct = 0
            total_n = 0
            for idx in range(0, len(lines), config.batch_size): # y, lines
                mini_batch = tokenizer(
                    lines[idx:idx + config.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                y = labels[idx:idx + config.batch_size]
                y = torch.LongTensor(y)
                y = y.to(device)

                x = mini_batch['input_ids']
                x = x.to(device)
                mask = mini_batch['attention_mask']
                mask = mask.to(device)

                # Take feed-forward
                y_hat = model(x, attention_mask=mask)[0]
                y_hats += [y_hat]

                correct, n = get_acc(y_hat, y)
                tot_correct += correct
                total_n += n            


            labels = torch.LongTensor(labels)
            labels = labels.to(device)

            y_hats = torch.cat(y_hats, dim=0)

            # y : must LongTensor (if float: meaning regression )
            if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hats, dim=-1) == labels).sum() / float(labels.size(0))
            else:
                accuracy = 0


            print('|accuracy| :', float(tot_correct/total_n) ,'|accuracy2| :', float(accuracy))
            
            with open(config.txt_fn, "a") as f:
                f.write('|accuracy| : {:.4f} |accuracy2| : {:.4f}\n\n'.format(
                float(tot_correct/total_n) , float(accuracy) ))


    
class BertTrainer(Trainer):

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader, index_to_label,
    ):
        train_engine = EngineForBert(
            EngineForBert.train, # function
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert( # trainer = Engine(train_step_function)
            EngineForBert.validate, # function
            model, crit, optimizer, scheduler, self.config
        )

        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1) # trainer.run(data, max_epochs=100)

                               
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        
    
        
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            EngineForBert.check_best, # function
        )
        
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            EngineForBert.check_best_acc, # function
        )
        
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            EngineForBert.save_model, # function
            train_engine, self.config, index_to_label,               # arguments
        )
        
        #### custom ####
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            Epoch_end.test, # function
            model, self.config, # arguments
        )     
        
        
        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

#         model.load_state_dict(validation_engine.best_model) 

        return validation_engine.best_model , validation_engine.best_model2, model





class Tester():

    def __init__(self, config):
        self.config = config

    def test(
        self,
        model, crit, optimizer, scheduler,
        train_loader, test_loader
    ):
        train_engine = EngineForBert(
            EngineForBert.train, # function
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert( # trainer = Engine(train_step_function)
            EngineForBert.validate, # function
            model, crit, optimizer, scheduler, self.config
        )

        EngineForBert.attach_test(
            validation_engine,
            verbose=self.config.verbose
        )

        validation_engine.run(test_loader, max_epochs=1)    
