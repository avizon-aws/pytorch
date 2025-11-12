import torch
import unittest

class TestQuantizeOp(unittest.TestCase):
    
    def test_basic_fp8_quantization(self):
        input_tensor = torch.randn(128, 128, dtype=torch.float32)
        output, scale = torch.quantize(
            input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            quant_type=0,
            rounding_mode=0
        )
        
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(scale.shape[0], (128 * 128 + 31) // 32)
    
    def test_mxfp8_quantization(self):
        input_tensor = torch.randn(1024, dtype=torch.float32)
        output, scale = torch.quantize(
            input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            quant_type=1,
            rounding_mode=0
        )
        
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape[0], (1024 + 31) // 32)
    
    def test_per_tensor_quantization(self):
        input_tensor = torch.randn(64, 64, dtype=torch.float32)
        output, scale = torch.quantize(
            input_tensor,
            block_size=64*64,
            dtype=torch.float8_e4m3fn,
            quant_type=0,
            rounding_mode=0
        )
        
        self.assertEqual(scale.shape[0], 1)
if __name__ == '__main__':
    unittest.main()