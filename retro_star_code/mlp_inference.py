{"payload":{"allShortcutsEnabled":true,"fileTree":{"retro_star_task/retro_star_code":{"items":[{"name":"__init__.py","path":"retro_star_task/retro_star_code/__init__.py","contentType":"file"},{"name":"mlp_inference.py","path":"retro_star_task/retro_star_code/mlp_inference.py","contentType":"file"},{"name":"mlp_policies.py","path":"retro_star_task/retro_star_code/mlp_policies.py","contentType":"file"},{"name":"mlp_train.py","path":"retro_star_task/retro_star_code/mlp_train.py","contentType":"file"},{"name":"smiles_to_fp.py","path":"retro_star_task/retro_star_code/smiles_to_fp.py","contentType":"file"},{"name":"value_mlp.py","path":"retro_star_task/retro_star_code/value_mlp.py","contentType":"file"}],"totalCount":6},"retro_star_task":{"items":[{"name":"files","path":"retro_star_task/files","contentType":"directory"},{"name":"retro_star_code","path":"retro_star_task/retro_star_code","contentType":"directory"},{"name":"README.md","path":"retro_star_task/README.md","contentType":"file"},{"name":"__init__.py","path":"retro_star_task/__init__.py","contentType":"file"},{"name":"backward_model.py","path":"retro_star_task/backward_model.py","contentType":"file"},{"name":"download_files.sh","path":"retro_star_task/download_files.sh","contentType":"file"},{"name":"file_names.py","path":"retro_star_task/file_names.py","contentType":"file"},{"name":"retro_star_inventory.py","path":"retro_star_task/retro_star_inventory.py","contentType":"file"},{"name":"test_molecules.py","path":"retro_star_task/test_molecules.py","contentType":"file"},{"name":"value_function.py","path":"retro_star_task/value_function.py","contentType":"file"}],"totalCount":10},"":{"items":[{"name":"retro_star_task","path":"retro_star_task","contentType":"directory"},{"name":"tests","path":"tests","contentType":"directory"},{"name":".flake8","path":".flake8","contentType":"file"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":".pre-commit-config.yaml","path":".pre-commit-config.yaml","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"}],"totalCount":7}},"fileTreeProcessingTime":5.558164,"foldersToFetch":[],"reducedMotionEnabled":"system","repo":{"id":634214238,"defaultBranch":"main","name":"pretrained-reaction-models","ownerLogin":"AustinT","currentUserCanPush":true,"isFork":false,"isEmpty":false,"createdAt":"2023-04-29T13:12:52.000+01:00","ownerAvatar":"https://avatars.githubusercontent.com/u/22754579?v=4","public":true,"private":false,"isOrgOwned":false},"refInfo":{"name":"main","listCacheKey":"v0:1683053963.0","canEdit":true,"refType":"branch","currentOid":"ff1b0510739cb3370afdf216ca448a91d6a36972"},"path":"retro_star_task/retro_star_code/mlp_inference.py","currentUser":{"id":119689089,"login":"is541","userEmail":"is541@cam.ac.uk"},"blob":{"rawLines":["from __future__ import print_function","import numpy as np","import torch","import torch.nn.functional as F","from rdchiral.main import rdchiralRunText","from .mlp_policies import load_parallel_model, preprocess","from collections import defaultdict","","","def merge(reactant_d):","    ret = []","    for reactant, l in reactant_d.items():","        ss, ts = zip(*l)","        ret.append((reactant, sum(ss), list(ts)[0]))","    reactants, scores, templates = zip(","        *sorted(ret, key=lambda item: item[1], reverse=True)","    )","    return list(reactants), list(scores), list(templates)","","","class MLPModel(object):","    def __init__(self, state_path, template_path, device=-1, fp_dim=2048):","        super(MLPModel, self).__init__()","        self.fp_dim = fp_dim","        self.net, self.idx2rules = load_parallel_model(","            state_path, template_path, fp_dim","        )","        self.net.eval()","        self.device = device","        if device >= 0:","            self.net.to(device)","","    def run(self, x, topk=10):","        arr = preprocess(x, self.fp_dim)","        arr = np.reshape(arr, [-1, arr.shape[0]])","        arr = torch.tensor(arr, dtype=torch.float32)","        if self.device >= 0:","            arr = arr.to(self.device)","        preds = self.net(arr)","        preds = F.softmax(preds, dim=1)","        if self.device >= 0:","            preds = preds.cpu()","        probs, idx = torch.topk(preds, k=topk)","        # probs = F.softmax(probs,dim=1)","        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]","        reactants = []","        scores = []","        templates = []","        for i, rule in enumerate(rule_k):","            out1 = []","            try:","                out1 = rdchiralRunText(rule, x)","                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))","                if len(out1) == 0:","                    continue","                # if len(out1) > 1: print(\"more than two reactants.\"),print(out1)","                out1 = sorted(out1)","                for reactant in out1:","                    reactants.append(reactant)","                    scores.append(probs[0][i].item() / len(out1))","                    templates.append(rule)","            # out1 = rdchiralRunText(x, rule)","            except ValueError:","                pass","        if len(reactants) == 0:","            return None","        reactants_d = defaultdict(list)","        for r, s, t in zip(reactants, scores, templates):","            if \".\" in r:","                str_list = sorted(r.strip().split(\".\"))","                reactants_d[\".\".join(str_list)].append((s, t))","            else:","                reactants_d[r].append((s, t))","","        reactants, scores, templates = merge(reactants_d)","        total = sum(scores)","        scores = [s / total for s in scores]","        return {\"reactants\": reactants, \"scores\": scores, \"template\": templates}","","","if __name__ == \"__main__\":","    import argparse","    from pprint import pprint","","    parser = argparse.ArgumentParser(description=\"Policies for retrosynthesis Planner\")","    parser.add_argument(","        \"--template_rule_path\",","        default=\"../data/uspto_all/template_rules_1.dat\",","        type=str,","        help=\"Specify the path of all template rules.\",","    )","    parser.add_argument(","        \"--model_path\",","        default=\"../model/saved_rollout_state_1_2048.ckpt\",","        type=str,","        help=\"specify where the trained model is\",","    )","    args = parser.parse_args()","    state_path = args.model_path","    template_path = args.template_rule_path","    model = MLPModel(state_path, template_path, device=-1)","    x = \"[F-:1]\"","    # x = '[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]'","    # x = '[S:3](=[O:4])(=[O:5])([O:6][CH2:7][CH:8]([CH2:9][CH2:10][CH2:11][CH3:12])[CH2:13][CH3:14])[OH:15]'","    # x = 'OCC(=O)OCCCO'","    # x = 'CC(=O)NC1=CC=C(O)C=C1'","    x = \"S=C(Cl)(Cl)\"","    # x = \"NCCNC(=O)c1ccc(/C=N/Nc2ncnc3c2cnn3-c2ccccc2)cc1\"","    # x = 'CCOC(=O)c1cnc2c(F)cc(Br)cc2c1O'","    # x = 'COc1cc2ncnc(Oc3cc(NC(=O)Nc4cc(C(C)(C)C(F)(F)F)on4)ccc3F)c2cc1OC'","    # x = 'COC(=O)c1ccc(CN2C(=O)C3(COc4cc5c(cc43)OCCO5)c3ccccc32)o1'","    x = \"O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3\"","    # x = 'CO[C@H](CC(=O)O)C(=O)O'","    # x = 'O=C(O)c1cc(OCC(F)(F)F)c(C2CC2)cn1'","    y = model.run(x, 10)","    pprint(y)"],"stylingDirectives":[[{"start":0,"end":4,"cssClass":"pl-k"},{"start":16,"end":22,"cssClass":"pl-k"},{"start":23,"end":37,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-k"},{"start":16,"end":18,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-s1"},{"start":16,"end":26,"cssClass":"pl-s1"},{"start":27,"end":29,"cssClass":"pl-k"},{"start":30,"end":31,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":13,"cssClass":"pl-s1"},{"start":14,"end":18,"cssClass":"pl-s1"},{"start":19,"end":25,"cssClass":"pl-k"},{"start":26,"end":41,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":6,"end":18,"cssClass":"pl-s1"},{"start":19,"end":25,"cssClass":"pl-k"},{"start":26,"end":45,"cssClass":"pl-s1"},{"start":47,"end":57,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-k"},{"start":24,"end":35,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":20,"cssClass":"pl-s1"}],[{"start":4,"end":7,"cssClass":"pl-s1"},{"start":8,"end":9,"cssClass":"pl-c1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":23,"end":33,"cssClass":"pl-s1"},{"start":34,"end":39,"cssClass":"pl-en"}],[{"start":8,"end":10,"cssClass":"pl-s1"},{"start":12,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":20,"cssClass":"pl-en"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":22,"end":23,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-en"},{"start":20,"end":28,"cssClass":"pl-s1"},{"start":30,"end":33,"cssClass":"pl-en"},{"start":34,"end":36,"cssClass":"pl-s1"},{"start":39,"end":43,"cssClass":"pl-en"},{"start":44,"end":46,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":15,"end":21,"cssClass":"pl-s1"},{"start":23,"end":32,"cssClass":"pl-s1"},{"start":33,"end":34,"cssClass":"pl-c1"},{"start":35,"end":38,"cssClass":"pl-en"}],[{"start":8,"end":9,"cssClass":"pl-c1"},{"start":9,"end":15,"cssClass":"pl-en"},{"start":16,"end":19,"cssClass":"pl-s1"},{"start":21,"end":24,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":25,"end":31,"cssClass":"pl-k"},{"start":32,"end":36,"cssClass":"pl-s1"},{"start":38,"end":42,"cssClass":"pl-s1"},{"start":43,"end":44,"cssClass":"pl-c1"},{"start":47,"end":54,"cssClass":"pl-s1"},{"start":54,"end":55,"cssClass":"pl-c1"},{"start":55,"end":59,"cssClass":"pl-c1"}],[],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":15,"cssClass":"pl-en"},{"start":16,"end":25,"cssClass":"pl-s1"},{"start":28,"end":32,"cssClass":"pl-en"},{"start":33,"end":39,"cssClass":"pl-s1"},{"start":42,"end":46,"cssClass":"pl-en"},{"start":47,"end":56,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":5,"cssClass":"pl-k"},{"start":6,"end":14,"cssClass":"pl-v"},{"start":15,"end":21,"cssClass":"pl-s1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":21,"cssClass":"pl-s1"},{"start":23,"end":33,"cssClass":"pl-s1"},{"start":35,"end":48,"cssClass":"pl-s1"},{"start":50,"end":56,"cssClass":"pl-s1"},{"start":56,"end":57,"cssClass":"pl-c1"},{"start":57,"end":58,"cssClass":"pl-c1"},{"start":58,"end":59,"cssClass":"pl-c1"},{"start":61,"end":67,"cssClass":"pl-s1"},{"start":67,"end":68,"cssClass":"pl-c1"},{"start":68,"end":72,"cssClass":"pl-c1"}],[{"start":8,"end":13,"cssClass":"pl-en"},{"start":14,"end":22,"cssClass":"pl-v"},{"start":24,"end":28,"cssClass":"pl-s1"},{"start":30,"end":38,"cssClass":"pl-en"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":28,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":16,"cssClass":"pl-s1"},{"start":18,"end":22,"cssClass":"pl-s1"},{"start":23,"end":32,"cssClass":"pl-s1"},{"start":33,"end":34,"cssClass":"pl-c1"},{"start":35,"end":54,"cssClass":"pl-en"}],[{"start":12,"end":22,"cssClass":"pl-s1"},{"start":24,"end":37,"cssClass":"pl-s1"},{"start":39,"end":45,"cssClass":"pl-s1"}],[],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":16,"cssClass":"pl-s1"},{"start":17,"end":21,"cssClass":"pl-en"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":28,"cssClass":"pl-s1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":17,"cssClass":"pl-s1"},{"start":18,"end":20,"cssClass":"pl-c1"},{"start":21,"end":22,"cssClass":"pl-c1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":20,"cssClass":"pl-s1"},{"start":21,"end":23,"cssClass":"pl-en"},{"start":24,"end":30,"cssClass":"pl-s1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":11,"cssClass":"pl-en"},{"start":12,"end":16,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":21,"end":25,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":26,"end":28,"cssClass":"pl-c1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":24,"cssClass":"pl-en"},{"start":25,"end":26,"cssClass":"pl-s1"},{"start":28,"end":32,"cssClass":"pl-s1"},{"start":33,"end":39,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":16,"cssClass":"pl-s1"},{"start":17,"end":24,"cssClass":"pl-en"},{"start":25,"end":28,"cssClass":"pl-s1"},{"start":31,"end":32,"cssClass":"pl-c1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":35,"end":38,"cssClass":"pl-s1"},{"start":39,"end":44,"cssClass":"pl-s1"},{"start":45,"end":46,"cssClass":"pl-c1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":19,"cssClass":"pl-s1"},{"start":20,"end":26,"cssClass":"pl-en"},{"start":27,"end":30,"cssClass":"pl-s1"},{"start":32,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":38,"end":43,"cssClass":"pl-s1"},{"start":44,"end":51,"cssClass":"pl-s1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":15,"cssClass":"pl-s1"},{"start":16,"end":22,"cssClass":"pl-s1"},{"start":23,"end":25,"cssClass":"pl-c1"},{"start":26,"end":27,"cssClass":"pl-c1"}],[{"start":12,"end":15,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":18,"end":21,"cssClass":"pl-s1"},{"start":22,"end":24,"cssClass":"pl-en"},{"start":25,"end":29,"cssClass":"pl-s1"},{"start":30,"end":36,"cssClass":"pl-s1"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":20,"cssClass":"pl-s1"},{"start":21,"end":24,"cssClass":"pl-en"},{"start":25,"end":28,"cssClass":"pl-s1"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":17,"cssClass":"pl-v"},{"start":18,"end":25,"cssClass":"pl-en"},{"start":26,"end":31,"cssClass":"pl-s1"},{"start":33,"end":36,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"},{"start":37,"end":38,"cssClass":"pl-c1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":15,"cssClass":"pl-s1"},{"start":16,"end":22,"cssClass":"pl-s1"},{"start":23,"end":25,"cssClass":"pl-c1"},{"start":26,"end":27,"cssClass":"pl-c1"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":25,"cssClass":"pl-s1"},{"start":26,"end":29,"cssClass":"pl-en"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":15,"end":18,"cssClass":"pl-s1"},{"start":19,"end":20,"cssClass":"pl-c1"},{"start":21,"end":26,"cssClass":"pl-s1"},{"start":27,"end":31,"cssClass":"pl-en"},{"start":32,"end":37,"cssClass":"pl-s1"},{"start":39,"end":40,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":41,"end":45,"cssClass":"pl-s1"}],[{"start":8,"end":40,"cssClass":"pl-c"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":18,"end":22,"cssClass":"pl-s1"},{"start":23,"end":32,"cssClass":"pl-s1"},{"start":33,"end":35,"cssClass":"pl-s1"},{"start":37,"end":40,"cssClass":"pl-k"},{"start":41,"end":43,"cssClass":"pl-s1"},{"start":44,"end":46,"cssClass":"pl-c1"},{"start":47,"end":50,"cssClass":"pl-s1"},{"start":51,"end":52,"cssClass":"pl-c1"},{"start":54,"end":59,"cssClass":"pl-en"},{"start":62,"end":68,"cssClass":"pl-en"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"}],[{"start":8,"end":11,"cssClass":"pl-k"},{"start":12,"end":13,"cssClass":"pl-s1"},{"start":15,"end":19,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":23,"end":32,"cssClass":"pl-en"},{"start":33,"end":39,"cssClass":"pl-s1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"}],[{"start":12,"end":15,"cssClass":"pl-k"}],[{"start":16,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":38,"cssClass":"pl-en"},{"start":39,"end":43,"cssClass":"pl-s1"},{"start":45,"end":46,"cssClass":"pl-s1"}],[{"start":16,"end":87,"cssClass":"pl-c"}],[{"start":16,"end":18,"cssClass":"pl-k"},{"start":19,"end":22,"cssClass":"pl-en"},{"start":23,"end":27,"cssClass":"pl-s1"},{"start":29,"end":31,"cssClass":"pl-c1"},{"start":32,"end":33,"cssClass":"pl-c1"}],[{"start":20,"end":28,"cssClass":"pl-k"}],[{"start":16,"end":81,"cssClass":"pl-c"}],[{"start":16,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":29,"cssClass":"pl-en"},{"start":30,"end":34,"cssClass":"pl-s1"}],[{"start":16,"end":19,"cssClass":"pl-k"},{"start":20,"end":28,"cssClass":"pl-s1"},{"start":29,"end":31,"cssClass":"pl-c1"},{"start":32,"end":36,"cssClass":"pl-s1"}],[{"start":20,"end":29,"cssClass":"pl-s1"},{"start":30,"end":36,"cssClass":"pl-en"},{"start":37,"end":45,"cssClass":"pl-s1"}],[{"start":20,"end":26,"cssClass":"pl-s1"},{"start":27,"end":33,"cssClass":"pl-en"},{"start":34,"end":39,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":43,"end":44,"cssClass":"pl-s1"},{"start":46,"end":50,"cssClass":"pl-en"},{"start":53,"end":54,"cssClass":"pl-c1"},{"start":55,"end":58,"cssClass":"pl-en"},{"start":59,"end":63,"cssClass":"pl-s1"}],[{"start":20,"end":29,"cssClass":"pl-s1"},{"start":30,"end":36,"cssClass":"pl-en"},{"start":37,"end":41,"cssClass":"pl-s1"}],[{"start":12,"end":45,"cssClass":"pl-c"}],[{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":29,"cssClass":"pl-v"}],[{"start":16,"end":20,"cssClass":"pl-k"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":14,"cssClass":"pl-en"},{"start":15,"end":24,"cssClass":"pl-s1"},{"start":26,"end":28,"cssClass":"pl-c1"},{"start":29,"end":30,"cssClass":"pl-c1"}],[{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":23,"cssClass":"pl-c1"}],[{"start":8,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":33,"cssClass":"pl-en"},{"start":34,"end":38,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-k"},{"start":12,"end":13,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":23,"end":26,"cssClass":"pl-en"},{"start":27,"end":36,"cssClass":"pl-s1"},{"start":38,"end":44,"cssClass":"pl-s1"},{"start":46,"end":55,"cssClass":"pl-s1"}],[{"start":12,"end":14,"cssClass":"pl-k"},{"start":15,"end":18,"cssClass":"pl-s"},{"start":19,"end":21,"cssClass":"pl-c1"},{"start":22,"end":23,"cssClass":"pl-s1"}],[{"start":16,"end":24,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":27,"end":33,"cssClass":"pl-en"},{"start":34,"end":35,"cssClass":"pl-s1"},{"start":36,"end":41,"cssClass":"pl-en"},{"start":44,"end":49,"cssClass":"pl-en"},{"start":50,"end":53,"cssClass":"pl-s"}],[{"start":16,"end":27,"cssClass":"pl-s1"},{"start":28,"end":31,"cssClass":"pl-s"},{"start":32,"end":36,"cssClass":"pl-en"},{"start":37,"end":45,"cssClass":"pl-s1"},{"start":48,"end":54,"cssClass":"pl-en"},{"start":56,"end":57,"cssClass":"pl-s1"},{"start":59,"end":60,"cssClass":"pl-s1"}],[{"start":12,"end":16,"cssClass":"pl-k"}],[{"start":16,"end":27,"cssClass":"pl-s1"},{"start":28,"end":29,"cssClass":"pl-s1"},{"start":31,"end":37,"cssClass":"pl-en"},{"start":39,"end":40,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-s1"}],[],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":19,"end":25,"cssClass":"pl-s1"},{"start":27,"end":36,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":39,"end":44,"cssClass":"pl-en"},{"start":45,"end":56,"cssClass":"pl-s1"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":19,"cssClass":"pl-en"},{"start":20,"end":26,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":27,"cssClass":"pl-s1"},{"start":28,"end":31,"cssClass":"pl-k"},{"start":32,"end":33,"cssClass":"pl-s1"},{"start":34,"end":36,"cssClass":"pl-c1"},{"start":37,"end":43,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":16,"end":27,"cssClass":"pl-s"},{"start":29,"end":38,"cssClass":"pl-s1"},{"start":40,"end":48,"cssClass":"pl-s"},{"start":50,"end":56,"cssClass":"pl-s1"},{"start":58,"end":68,"cssClass":"pl-s"},{"start":70,"end":79,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":2,"cssClass":"pl-k"},{"start":3,"end":11,"cssClass":"pl-s1"},{"start":12,"end":14,"cssClass":"pl-c1"},{"start":15,"end":25,"cssClass":"pl-s"}],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":19,"cssClass":"pl-s1"}],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":15,"cssClass":"pl-s1"},{"start":16,"end":22,"cssClass":"pl-k"},{"start":23,"end":29,"cssClass":"pl-s1"}],[],[{"start":4,"end":10,"cssClass":"pl-s1"},{"start":11,"end":12,"cssClass":"pl-c1"},{"start":13,"end":21,"cssClass":"pl-s1"},{"start":22,"end":36,"cssClass":"pl-v"},{"start":37,"end":48,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"},{"start":49,"end":86,"cssClass":"pl-s"}],[{"start":4,"end":10,"cssClass":"pl-s1"},{"start":11,"end":23,"cssClass":"pl-en"}],[{"start":8,"end":30,"cssClass":"pl-s"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":56,"cssClass":"pl-s"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":13,"end":16,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":13,"end":54,"cssClass":"pl-s"}],[],[{"start":4,"end":10,"cssClass":"pl-s1"},{"start":11,"end":23,"cssClass":"pl-en"}],[{"start":8,"end":22,"cssClass":"pl-s"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":58,"cssClass":"pl-s"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":13,"end":16,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":13,"end":49,"cssClass":"pl-s"}],[],[{"start":4,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":17,"cssClass":"pl-s1"},{"start":18,"end":28,"cssClass":"pl-en"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":21,"cssClass":"pl-s1"},{"start":22,"end":32,"cssClass":"pl-s1"}],[{"start":4,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-s1"},{"start":25,"end":43,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":20,"cssClass":"pl-v"},{"start":21,"end":31,"cssClass":"pl-s1"},{"start":33,"end":46,"cssClass":"pl-s1"},{"start":48,"end":54,"cssClass":"pl-s1"},{"start":54,"end":55,"cssClass":"pl-c1"},{"start":55,"end":56,"cssClass":"pl-c1"},{"start":56,"end":57,"cssClass":"pl-c1"}],[{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":8,"end":16,"cssClass":"pl-s"}],[{"start":4,"end":96,"cssClass":"pl-c"}],[{"start":4,"end":109,"cssClass":"pl-c"}],[{"start":4,"end":24,"cssClass":"pl-c"}],[{"start":4,"end":33,"cssClass":"pl-c"}],[{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":8,"end":21,"cssClass":"pl-s"}],[{"start":4,"end":59,"cssClass":"pl-c"}],[{"start":4,"end":42,"cssClass":"pl-c"}],[{"start":4,"end":75,"cssClass":"pl-c"}],[{"start":4,"end":68,"cssClass":"pl-c"}],[{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":8,"end":45,"cssClass":"pl-s"}],[{"start":4,"end":34,"cssClass":"pl-c"}],[{"start":4,"end":45,"cssClass":"pl-c"}],[{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":17,"cssClass":"pl-en"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":21,"end":23,"cssClass":"pl-c1"}],[{"start":4,"end":10,"cssClass":"pl-en"},{"start":11,"end":12,"cssClass":"pl-s1"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":null,"configFilePath":null,"networkDependabotPath":"/AustinT/pretrained-reaction-models/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":false,"repoAlertsPath":"/AustinT/pretrained-reaction-models/security/dependabot","repoSecurityAndAnalysisPath":"/AustinT/pretrained-reaction-models/settings/security_analysis","repoOwnerIsOrg":false,"currentUserCanAdminRepo":false},"displayName":"mlp_inference.py","displayUrl":"https://github.com/AustinT/pretrained-reaction-models/blob/main/retro_star_task/retro_star_code/mlp_inference.py?raw=true","headerInfo":{"blobSize":"4.15 KB","deleteInfo":{"deletePath":"https://github.com/AustinT/pretrained-reaction-models/delete/main/retro_star_task/retro_star_code/mlp_inference.py","deleteTooltip":"Delete this file"},"editInfo":{"editTooltip":"Edit this file"},"ghDesktopPath":"x-github-client://openRepo/https://github.com/AustinT/pretrained-reaction-models?branch=main&filepath=retro_star_task%2Fretro_star_code%2Fmlp_inference.py","gitLfsPath":null,"onBranch":true,"shortPath":"10654f9","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2FAustinT%2Fpretrained-reaction-models%2Fblob%2Fmain%2Fretro_star_task%2Fretro_star_code%2Fmlp_inference.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"116","truncatedSloc":"107"},"mode":"file"},"image":false,"isCodeownersFile":null,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","large":false,"loggedIn":true,"newDiscussionPath":"/AustinT/pretrained-reaction-models/discussions/new","newIssuePath":"/AustinT/pretrained-reaction-models/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/AustinT/pretrained-reaction-models/blob/main/retro_star_task/retro_star_code/mlp_inference.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/AustinT/pretrained-reaction-models/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"AustinT","repoName":"pretrained-reaction-models","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":null,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"merge","kind":"function","identStart":244,"identEnd":249,"extentStart":240,"extentEnd":561,"fullyQualifiedName":"merge","identUtf16":{"start":{"lineNumber":9,"utf16Col":4},"end":{"lineNumber":9,"utf16Col":9}},"extentUtf16":{"start":{"lineNumber":9,"utf16Col":0},"end":{"lineNumber":17,"utf16Col":57}}},{"name":"MLPModel","kind":"class","identStart":570,"identEnd":578,"extentStart":564,"extentEnd":2819,"fullyQualifiedName":"MLPModel","identUtf16":{"start":{"lineNumber":20,"utf16Col":6},"end":{"lineNumber":20,"utf16Col":14}},"extentUtf16":{"start":{"lineNumber":20,"utf16Col":0},"end":{"lineNumber":77,"utf16Col":80}}},{"name":"__init__","kind":"function","identStart":596,"identEnd":604,"extentStart":592,"extentEnd":953,"fullyQualifiedName":"MLPModel.__init__","identUtf16":{"start":{"lineNumber":21,"utf16Col":8},"end":{"lineNumber":21,"utf16Col":16}},"extentUtf16":{"start":{"lineNumber":21,"utf16Col":4},"end":{"lineNumber":30,"utf16Col":31}}},{"name":"run","kind":"function","identStart":963,"identEnd":966,"extentStart":959,"extentEnd":2819,"fullyQualifiedName":"MLPModel.run","identUtf16":{"start":{"lineNumber":32,"utf16Col":8},"end":{"lineNumber":32,"utf16Col":11}},"extentUtf16":{"start":{"lineNumber":32,"utf16Col":4},"end":{"lineNumber":77,"utf16Col":80}}}]}},"copilotUserAccess":{"canModifyCopilotSettings":false,"canViewCopilotSettings":false,"accessAllowed":false,"hasCFIAccess":true,"hasSubscriptionEnded":false,"business":null},"csrf_tokens":{"/AustinT/pretrained-reaction-models/branches":{"post":"Je0UQ07p9ryh1mgS7T021abQokcCXixiotVlHk0eDGKl9y-DGQ4FTdx8fyyVUaGEl7I5FDgaYAFXGMLdOCEAFg"}}},"title":"pretrained-reaction-models/retro_star_task/retro_star_code/mlp_inference.py at main · AustinT/pretrained-reaction-models","locale":"en"}