import paddle
import sys
sys.path.insert(0, '/home/cao/code/github/PaddleFormers')

from paddleformers.transformers import Phi4Config
from paddleformers.transformers import (
    Phi4Model, 
    Phi4ForCausalLM,
    Phi4Attention,
    Phi4Mamba,
    Phi4Cache,
    Phi4MLP,
    Phi4RMSNorm
)

def test_config():
    config = Phi4Config()
    print(f"Config created successfully")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  mb_per_layer: {config.mb_per_layer}")
    print(f"  mamba_d_state: {config.mamba_d_state}")
    return config

def test_components(config):
    batch_size = 2
    seq_len = 8
    
    print("\n=== Testing Phi4RMSNorm ===")
    try:
        norm = Phi4RMSNorm(config.hidden_size, eps=1e-5)
        x = paddle.randn([batch_size, seq_len, config.hidden_size])
        out = norm(x)
        print(f"  Input shape: {x.shape}, Output shape: {out.shape}")
        print(f"  ✓ Phi4RMSNorm works correctly")
    except Exception as e:
        print(f"  ✗ Phi4RMSNorm error: {e}")
    
    print("\n=== Testing Phi4MLP ===")
    try:
        mlp = Phi4MLP(config)
        x = paddle.randn([batch_size, seq_len, config.hidden_size])
        out = mlp(x)
        print(f"  Input shape: {x.shape}, Output shape: {out.shape}")
        print(f"  ✓ Phi4MLP works correctly")
    except Exception as e:
        print(f"  ✗ Phi4MLP error: {e}")
    
    print("\n=== Testing Phi4Cache ===")
    try:
        cache = Phi4Cache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=128,
            dtype=paddle.float32
        )
        print(f"  Cache layers: {len(cache.key_cache)}")
        print(f"  Global attention index: {cache.global_attn_idx}")
        print(f"  ✓ Phi4Cache created successfully")
    except Exception as e:
        print(f"  ✗ Phi4Cache error: {e}")
    
    print("\n=== Testing Phi4Attention ===")
    try:
        attn = Phi4Attention(config, layer_idx=1)
        x = paddle.randn([batch_size, seq_len, config.hidden_size])
        attn_output, attn_weights, yoco_kv = attn(x)
        print(f"  Input shape: {x.shape}, Output shape: {attn_output.shape}")
        print(f"  ✓ Phi4Attention works correctly")
    except Exception as e:
        print(f"  ✗ Phi4Attention error: {e}")

def test_model(config):
    print("\n=== Testing Phi4Model ===")
    try:
        model = Phi4Model(config)
        batch_size = 2
        seq_len = 8
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_len])
        
        outputs = model(input_ids=input_ids, use_cache=False)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        print(f"  ✓ Phi4Model forward pass successful")
    except Exception as e:
        print(f"  ✗ Phi4Model error: {e}")

def test_causal_lm(config):
    print("\n=== Testing Phi4ForCausalLM ===")
    try:
        model = Phi4ForCausalLM(config)
        batch_size = 2
        seq_len = 8
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_len])
        
        outputs = model(input_ids=input_ids, use_cache=False)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  ✓ Phi4ForCausalLM forward pass successful")
    except Exception as e:
        print(f"  ✗ Phi4ForCausalLM error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phi4 Model Optimization")
    print("=" * 60)
    
    config = test_config()
    test_components(config)
    test_model(config)
    test_causal_lm(config)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("All basic tests completed. Check output above for details.")
