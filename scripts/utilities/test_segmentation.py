from scripts.metrics import compute_iou
from torchvision.transforms import functional as TF
from IPython.display import display, HTML
from io import BytesIO
from base64 import b64encode

def test_IoU_Image(model,input,truth,ds_split,showImage=False):
    # Push through our network
    model = model.cpu()
    # model = model.gpu()
    output = model(input.unsqueeze(0))

    # Display the input, output and truth tensors
    template_table = '<table><thead><tr><th>Tensor</th><th>Shape</th><th>Image</th></tr></thead><tbody>{0}</tbody></table>'
    template_row = '<tr><td>{0}</td><td>{1}</td><td><img src="data:image/png;base64,{2}"/></td></tr>'

    input_img = TF.to_pil_image(input)
    output_img = ds_split["val"].to_image(ds_split["val"].masks_to_indices(output).squeeze(0))
    truth_img = ds_split["val"].to_image(truth)
    iou=compute_iou(output,truth)
    if showImage:
        rows = []
        for name, tensor, img in [('Input', input, input_img), ('Output', output, output_img), ('Target', truth, truth_img)]:
            with BytesIO() as b: 
                img.save(b, format='png')
                rows.append(template_row.format(name, list(tensor.shape), b64encode(b.getvalue()).decode('utf-8')))

        # Render HTML table
        table = template_table.format(''.join(rows))
        display(HTML(table))
    return output,iou