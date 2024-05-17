import os
import torch
from inference import inference_with_lora


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:6",
        # path setting
        "image_size": 256,
        "prompts_file": "./CheckpointStyleDataset/prompts.csv",
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "LoRAModel_path": "./CheckpointGenLoRA/class000",
        "save_sampled_images_path": "../../datasets/Generated/GenStyles/style00",
        # generating setting
        "batch_size": 100,
        "total_number": 10000,
        "dtype": torch.float16,
    }

    for i in range(0, 1000, 1):
        print("\nstart:", str(i).zfill(3))
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"

        for k in range(0, config["total_number"], config["batch_size"]):
            prompts = ["A bird landing on a perch by a feeder.",
                       "A man riding a skateboard down the side of a ramp.",
                       "A group of people standing in muddy water.",
                       "A baseball player holding a baseball bat during a game.",
                       "A bed sitting under a bedroom window next to a table.",
                       "A bird perched on top of a tree branch.",
                       "A blender is filled with various fruits and vegetables.",
                       "A boy in blue jeans rides his skateboard in crouched position.",
                       "A child in cold gear skiing on snow.",
                       "A child looks up at the woman bringing a birthday cake.",
                       "A city bus pulled up to the curb of a street",
                       "A close up of pastries in a display case.",
                       "A decorated room with a circular overhead light and a black couch.",
                       "A dog looks in from an open window with his paws on the windowsill.",
                       "a dog running in a room whild holding onto a soccer ball",
                       "a large intersection and a red fire hydrant sits on the corner",
                       "a little girl stands in front of a area with some giraffes in it",
                       "a man paddleboarding in the water by himself",
                       "A motor bike is shown parked on the road.",
                       "A passenger train that is going under a walking bridge.",
                       "A person in black gear heading down a snowy ski slope",
                       "A person water skiing on one ski down a river.",
                       "A person with an umbrella walking down a street.",
                       "a small toy train is on some tracks",
                       "A women who is sitting on some luggage.",
                       "A train traveling down tracks near power poles.",
                       "The brown horse looks over the black wall ahead.",
                       "Three giraffes in a fence, one of them bending down.",
                       "Two large cows and one small cow standing in a field",
                       "The man in the suit and tie is looking at dress shoes.",]
            # these prompts looks better
            print("\ngenerating: ", prompts)

            # inference
            images = inference_with_lora(prompt=prompts,
                                         lora_path=config["LoRAModel_path"],
                                         model_path=config["BaseModel_path"],
                                         dtype=config["dtype"],
                                         device=config["device"])

            # save
            save_folder = config["save_sampled_images_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=False)
                try:  # A group of people in the mountains walking/skiing.jpg
                    image.save(os.path.join(save_folder, prompt+".jpg"))
                except FileNotFoundError:
                    continue

