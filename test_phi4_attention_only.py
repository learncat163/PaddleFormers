import paddle
import sys
sys.path.insert(0, '/home/cao/code/github/PaddleFormers')

from paddleformers.transformers.phi4.configuration import Phi4Config
from paddleformers.transformers.phi4.modeling import Phi4Model, Phi4ForCausalLM

def test_attention_only_model():
    config = Phi4Config(mb_per_layer=0)
    print("=" * 60)
    print("Testing Phi4 Model (Attention Only - No Mamba)")
    print("=" * 60)
    print(f"Config: mb_per_layer={config.mb_per_layer}")
    
    print("\n=== Testing Phi4Model (Attention Only) ===")
    try:
        model = Phi4Model(config)
        batch_size = 2
        seq_len = 8
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_len])
        
        outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
        print(f"  ✓ Input shape: {input_ids.shape}")
        print(f"  ✓ Output shape: {outputs.last_hidden_state.shape}")
        print(f"  ✓ Phi4Model forward pass successful!")
    except Exception as e:
        print(f"  ✗ Phi4Model error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Phi4ForCausalLM (Attention Only) ===")
    try:
        model = Phi4ForCausalLM(config)
        batch_size = 2
        seq_len = 8
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_len])
        
        outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
        print(f"  ✓ Input shape: {input_ids.shape}")
        print(f"  ✓ Logits shape: {outputs.logits.shape}")
        expected_shape = [batch_size, seq_len, config.vocab_size]
        assert list(outputs.logits.shape) == expected_shape, f"Expected {expected_shape}, got {list(outputs.logits.shape)}"
        print(f"  ✓ Phi4ForCausalLM forward pass successful!")
    except Exception as e:
        print(f"  ✗ Phi4ForCausalLM error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attention_only_model()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
