"""Bounded forward hooks on real activations; no recomputed attention or fake nodes."""
from contextlib import AbstractContextManager
from time import perf_counter
import numpy as np


class NeuralCapture(AbstractContextManager):
    def __init__(self,model,branch,token_labels,trace,on_event=None):
        self.model=model;self.branch=branch;self.labels=token_labels
        self.trace=trace;self.on_event=on_event;self.handles=[];self.started=perf_counter()
        self.frames=[]

    def __enter__(self):
        if self.branch=='E2':
            layers=self.model.bert.encoder.layer
            modules=[('词嵌入 + 位置编码',self.model.bert.embeddings,'embedding',0)]
            for i,layer in enumerate(layers,1):
                modules.extend([(f'第 {i} 层 · 注意力输出',layer.attention,'attention_output',i),
                                (f'第 {i} 层 · FFN 激活',layer.intermediate,'feed_forward',i),
                                (f'第 {i} 层 · 残差归一化输出',layer,'layer_output',i)])
        else:
            layers=self.model.layers
            modules=[('词嵌入',self.model.embed_tokens,'embedding',0)]
            for i,layer in enumerate(layers,1):
                modules.extend([(f'第 {i} 层 · 注意力输出',layer.self_attn,'attention_output',i),
                                (f'第 {i} 层 · 门控 FFN 输出',layer.mlp,'feed_forward',i),
                                (f'第 {i} 层 · 残差输出',layer,'layer_output',i)])
            modules.append(('最终归一化',self.model.norm,'final_norm',len(layers)))
        self.trace['neural_capture']={'frames':self.frames,'layer_count':len(layers),
            'sample_rule':'up to 16 evenly spaced token positions and 16 evenly spaced hidden/FFN dimensions; final token always included',
            'kind':'actual_forward_hooks','attention_weights_captured':False}
        try:
            def inputs_hook(module,args,kwargs):
                ids=kwargs.get('input_ids',args[0] if args else None)
                if ids is not None:self.trace['model_input_ids']=ids.detach()[0].cpu().tolist()
            self.handles.append(self.model.register_forward_pre_hook(inputs_hook,with_kwargs=True))
            for title,module,kind,index in modules:
                self.handles.append(module.register_forward_hook(self._hook(title,kind,index)))
            if self.branch=='E2':
                def logits_hook(module,args,output):
                    self.trace['class_logits']=output.detach()[0].float().cpu().tolist()
                self.handles.append(self.model.classifier.register_forward_hook(logits_hook))
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def _hook(self,title,kind,index):
        def capture(module,args,output):
            tensor=output[0] if isinstance(output,(tuple,list)) else output
            if tensor is None or tensor.ndim!=3:return
            # One mail, no padded companion rows. Sampling is index-based, never
            # selected for dramatic activation or a preferred classification.
            tokens=np.linspace(0,tensor.shape[1]-1,min(16,tensor.shape[1]),dtype=int).tolist()
            dims=np.linspace(0,tensor.shape[2]-1,min(16,tensor.shape[2]),dtype=int).tolist()
            sample=tensor.detach()[0,tokens][:,dims].float().cpu().numpy()
            frame={'title':title,'kind':kind,'layer':index,'shape':list(tensor.shape),
                   'token_indices':tokens,'token_labels':[self.labels[i] for i in tokens],
                   'dimensions':dims,'values':sample.tolist(),
                   'sample_rms':float(np.sqrt(np.mean(sample**2))),
                   'elapsed_ms':(perf_counter()-self.started)*1000}
            self.frames.append(frame)
            if self.on_event:self.on_event({'branch':self.branch,'phase':'forward','title':title,
                                           'layer':index,'layers':self.trace['neural_capture']['layer_count'],
                                           'sample_rms':frame['sample_rms'],'frame':frame})
        return capture

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if self.frames:
            scale=max(abs(v) for f in self.frames for row in f['values'] for v in row)
            self.trace['neural_capture']['global_abs_scale']=scale or 1.0
        return False
