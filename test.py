import torch

batch_size = 1
num_anchors = 3
num_classes = 20
grid = 13

tx = torch.zeros((batch_size, num_anchors, grid, grid))
ty = torch.zeros((batch_size, num_anchors, grid, grid))
tw = torch.zeros((batch_size, num_anchors, grid, grid))
th = torch.zeros((batch_size, num_anchors, grid, grid))
tconf = torch.zeros((batch_size, num_anchors, grid, grid))
tcls = torch.zeros((batch_size, num_anchors, grid, grid, num_classes))

tx[:, 1, 3, 3] = 0.1
ty[:, 1, 3, 3] = 0.2
tw[:, 1, 3, 3] = 0.3
th[:, 1, 3, 3] = 0.4
tconf[:, 1, 3, 3] = 0.5
tcls[:, 1, 3, 3, 0] = 1

tx = tx.unsqueeze(-1)
ty = ty.unsqueeze(-1)
tw = tw.unsqueeze(-1)
th = th.unsqueeze(-1)
tconf = tconf.unsqueeze(-1)

output = torch.cat([tx, ty, tw, th, tconf, tcls], -1) # [batch_size, num_anchors, layer_h, layer_w, 5(tx, ty, tw, th, tconf) + num_classes]
print(f'{output[:, 1, 3, 3, :]}')
print(f'{output[..., 4].sigmoid()}')
print(f'{output[..., 4:5].sigmoid()}')

output = output.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, -1, grid, grid) # [batch_size, num_anchors*(5(tx, ty, tw, th, tconf) + num_classes), layer_h, layer_w]

predictions = output.view(batch_size, num_anchors, (5+num_classes), grid, grid).permute(0, 1, 3, 4, 2).contiguous()
print(f'{predictions[:, 1, 3, 3, :]}')