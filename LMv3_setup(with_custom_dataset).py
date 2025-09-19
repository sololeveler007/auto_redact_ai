from transformers import AutoProcessor, TrainingArguments, Trainer, LayoutLMv3ForTokenClassification
from datasets import load_dataset, Features, Array2D, Array3D, Value, Sequence
import torch
from evaluate import load
import numpy as np
from transformers.data.data_collator import default_data_collator
import traceback

def use_LayoutLMv3():
    
    try:
        configured = False

        data_set = load_dataset('dark1007/pii_dataset')  #* loading dataset
        labels_names = ['O','B-NAME','I-NAME','B-ADDRESS','I-ADDRESS','B-ID','I-ID','B-EMAIL','I-EMAIL','B-PHONE','I-PHONE','B-DOB','I-DOB']
        label2id = {label: i for i,label in enumerate(labels_names)}  #* label2id
        id2label = {i: label for i,label in enumerate(labels_names)}  #* id2label
        num_labels = len(labels_names)

        while not configured:
            try:
                processor = AutoProcessor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr = False) #* AutoProcessor
                model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base', id2label= id2label, label2id= label2id) #*model creation
                configured = True
            except Exception as e:
                _ = ''
                # print("Error:",str(e))

        #* mapping function
        def encoding_for_data(example): 
            images = example['image']
            tokens = example['tokens']
            bboxes = example['bboxes']
            word_labels = example['ner_tags']
            
            encodings = processor(images, text = tokens, boxes = bboxes, word_labels = word_labels,
                                  truncation = True, padding= 'max_length')
            # print(list(encodings.keys()))
            return encodings
        
        #* features instance creation
        features = Features({
            "input_ids" : Sequence(Value(dtype="int64")),
            "attention_mask" : Sequence(Value(dtype="int64")),
            "bbox" : Array2D(dtype="int64",shape=(512,4)),
            "labels" : Sequence(Value("int64")),
            "pixel_values" : Array3D(dtype="float32",shape=(3,224,224))
        })

        #* selecting sample data range
        train_sample = data_set['train'].shuffle(seed=42) #* train_sample
        test_sample = data_set['test'].shuffle(seed=42)   #* test_sample
        
        column_names = train_sample.column_names
        train_dataset = train_sample.map(encoding_for_data, batched = True, remove_columns=column_names, features=features)  #* model train inputs
        eval_dataset = test_sample.map(encoding_for_data, batched = True, remove_columns=column_names, features=features)   #* model test inputs
        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")

        metric = load("seqeval")

        return_entity_level_metrics = True

        #* evauation metric function
        def evaluate_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            #* remove paddded values from predictions and labels
            prediction_labels = [
                [labels_names[p] for (p, l) in zip(prediction , label) if l != -100]
                for (prediction,label) in zip(predictions, labels)
            ]

            true_labels = [
                [labels_names[l] for (p, l) in zip(prediction , label) if l != -100]
                for (prediction,label) in zip(predictions, labels)
            ]

            results = metric.compute(predictions = prediction_labels, references = true_labels)
            final_result = {}
            if return_entity_level_metrics:
                for key, value in results.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            final_result[f"{key}_{k}"] = v
                    else:
                        final_result[key] = value
                return final_result
            else:
                return {
                    "overall_precision" : results['overall_precision'],
                    "overall_recall" : results['overall_recall'],
                    "overall_f1" : results['overall_f1'],
                    "overall_accuracy" : results['overall_accuracy']
                }

        #* training arguments
        # help(TrainingArguments)

        training_arguments = TrainingArguments(
            output_dir = './custom_model_configs/layoutlmv3-finetuned-cord_100',
            max_steps = 1000,
            eval_strategy = 'steps',
            eval_steps = 250,
            per_device_train_batch_size = 4,
            per_device_eval_batch_size= 4,
            load_best_model_at_end = True,
            metric_for_best_model = 'overall_accuracy',
            learning_rate = 1e-5
        )
        # help(Trainer)
        #* trainer setup
        trainer = Trainer(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processor,
            compute_metrics = evaluate_metrics,
            args = training_arguments,
            data_collator= default_data_collator

        )
        #* training and evaluation
        try:
            trainer.train()
            trainer.evaluate()
        except Exception as e:
            print("Error:",str(e))
            traceback.print_exc()

        model.save_pretrained('./custom_model_configs/layoutlmv3-finetuned-custom')
        processor.save_pretrained('./custom_model_configs/layoutlmv3-finetuned-custom_process')
        print("executed")




    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    pass