(function(){"use strict";const k=`
flow_lm:
  dtype: float32
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3

mimi:
  dtype: float32
  sample_rate: 24000
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    channels: 1
    n_filters: 64
    n_residual_layers: 1
    ratios: [6, 5, 4]
    kernel_size: 7
    residual_kernel_size: 3
    last_kernel_size: 3
    dilation_base: 2
    pad_mode: constant
    compress: 2
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 2
    layer_scale: 0.01
    context: 250
    dim_feedforward: 2048
    input_dimension: 512
    output_dimensions: [512]
  quantizer:
    dimension: 32
    output_dimension: 512
`,g=["alba","marius","javert","jean","fantine","cosette","eponine","azelma"],b=new TextEncoder;let f=null,c=null,m=24e3,l=!1,d=0;const i=(t,e=[])=>{self.postMessage(t,e)},h=t=>{i({kind:"status",status:t})},u=(t,e)=>{i({kind:"rpc_ok",requestId:t,payload:e})},v=(t,e)=>{const r=e instanceof Error?e.message:String(e);i({kind:"rpc_err",requestId:t,error:r})},E=t=>new Promise(e=>setTimeout(e,t)),S=t=>g.includes(t),y=()=>{if(!c?.is_ready())throw new Error("WASM model is not initialized yet.");return c},M=async(t,e)=>{const r="/tts_b6369a24.safetensors";try{const n=await fetch(r);if(n.ok)return{bytes:new Uint8Array(await n.arrayBuffer()),source:"local"}}catch{}if(!t.trim())throw new Error("HF repository is required (e.g. kyutai/pocket-tts)");const s={};e.trim()&&(s.Authorization=`Bearer ${e.trim()}`);const a=`https://huggingface.co/${t.trim()}/resolve/main/tts_b6369a24.safetensors`,o=await fetch(a,{headers:s});if(!o.ok)throw o.status===401?new Error("HF auth required (token missing or invalid for gated repo)"):new Error(`Failed to fetch model weights (${o.status})`);return{bytes:new Uint8Array(await o.arrayBuffer()),source:"hf"}},A=async(t,e,r)=>{const s=`/embeddings/${t}.safetensors`;try{const _=await fetch(s);if(_.ok)return new Uint8Array(await _.arrayBuffer())}catch{}if(!e.trim())throw new Error("HF repository is required for preset voices in WASM mode.");const a={};r.trim()&&(a.Authorization=`Bearer ${r.trim()}`);const o=`https://huggingface.co/${e.trim()}/resolve/main/embeddings/${t}.safetensors`,n=await fetch(o,{headers:a});if(!n.ok)throw n.status===401?new Error("HF auth required for preset voice fetch."):new Error(`Failed to fetch preset voice (${n.status})`);return new Uint8Array(await n.arrayBuffer())},B=async t=>{l=!0,d+=1,h({phase:"initializing-runtime",progress:10,message:"Loading WASM runtime...",source:null,ready:!1,error:null}),f||(f=await import(`${t.wasmBase.replace(/\/+$/,"")}/pocket_tts.js`)),await f.default();const e=t.manualAssets;let r="manual",s=e?.weightsBytes;if(!s){h({phase:"loading-assets",progress:42,message:"Fetching model weights...",source:null,ready:!1,error:null});const n=await M(t.hfRepo,t.hfToken);s=n.bytes,r=n.source}const a=e?.configBytes??b.encode(k),o=e?.tokenizerBytes??new Uint8Array(0);h({phase:"compiling-model",progress:78,message:"Compiling model in WASM...",source:r,ready:!1,error:null}),c=new f.WasmTTSModel,c.load_from_buffer(a,s,o),m=c.sample_rate,h({phase:"ready",progress:100,message:"WASM model is ready.",source:r,ready:!0,error:null}),u(t.requestId,{sampleRate:m})},z=async t=>{const e=y(),r=t.voice;if(r.kind==="wav"){e.load_voice_from_buffer(r.wavBytes),u(t.requestId);return}if(r.kind==="embedding"){e.load_voice_from_safetensors(r.embeddingBytes),u(t.requestId);return}if(!S(r.voice))throw new Error(`Unknown preset voice: ${r.voice}`);const s=await A(r.voice,r.hfRepo,r.hfToken);e.load_voice_from_safetensors(s),u(t.requestId)},q=async t=>{const e=y();l=!1;const r=++d,s=e.start_stream(t.text);let a=!1,o=0;for(;!l&&r===d;){const n=Math.max(320,Math.floor(m*.032)),_=Math.max(1024,Math.floor(m*.11)),$=o<3?n:_,w=s.next_chunk_min_samples($);if(w==null)break;a||(a=!0,i({kind:"stream_first_chunk"}));const p=s.last_chunk_stats(),F=typeof p.compute_ms=="number"?p.compute_ms:null,I=typeof p.chunks_merged=="number"?p.chunks_merged:null;i({kind:"stream_chunk",chunk:w,computeMs:F,mergedChunks:I},[w.buffer]),o+=1,o%6===0&&await E(0)}if(l||r!==d)throw new Error("abort");i({kind:"stream_done"}),u(t.requestId)};self.onmessage=t=>{const e=t.data;if(!(!e||typeof e!="object")){if(e.kind==="stop"){l=!0,d+=1;return}(async()=>{try{if(e.kind==="init"){await B(e);return}if(e.kind==="prepare_voice"){await z(e);return}if(e.kind==="start_stream"){await q(e);return}}catch(r){if(e.kind==="start_stream"){const s=r instanceof Error?r.message:String(r);i({kind:"stream_error",error:s})}v(e.requestId,r)}})()}}})();
