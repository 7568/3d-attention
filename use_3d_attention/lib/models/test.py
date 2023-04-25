import torch
from einops import rearrange


x2 = torch.randn(2,3,4,5)
x3 = rearrange(x2, 'b s d f -> b (s d) f')
x4 = rearrange(x3, 'b (s d) f -> b d s f',s=3)
x5 = rearrange(x3, 'b (s d) f -> b s d f',s=3)
x5 = x5.permute(0,2,1, 3)
print(torch.eq(x4,x5).any())

print([[1,2] for i in range(10)])