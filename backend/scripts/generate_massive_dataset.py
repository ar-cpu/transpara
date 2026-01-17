import random
import itertools

extreme_left = {
    'subjects': ['workers', 'the proletariat', 'labor', 'the working class', 'employees', 'wage earners'],
    'actions': ['must seize', 'deserve to control', 'should own', 'need to take', 'have the right to'],
    'objects': ['the means of production', 'factories and land', 'industry', 'capital', 'productive resources'],
    'modifiers': ['from the capitalist class', 'from billionaires', 'from the bourgeoisie', 'from exploiters', 'from owners'],
}

left = {
    'policies': ['universal healthcare', 'free college', 'affordable housing', 'living wages', 'union rights'],
    'verbs': ['ensures', 'guarantees', 'provides', 'protects', 'supports'],
    'benefits': ['everyone can access treatment', 'all families thrive', 'communities prosper', 'people live with dignity', 'children succeed'],
}

center_left = {
    'approach': ['evidence based policy', 'pragmatic reform', 'smart regulation', 'balanced solutions', 'data driven approach'],
    'combines': ['market efficiency and social goals', 'growth and fairness', 'innovation and protection', 'business and workers'],
}

center = {
    'intro': ['the data shows', 'studies indicate', 'research suggests', 'analysts report', 'experts note'],
    'stat': ['unemployment at 3.7 percent', 'gdp growth of 2.5 percent', 'inflation rose to 4.2 percent', 'rates increased by 0.25 points'],
    'conclusion': ['according to recent data', 'based on historical trends', 'as economists predicted', 'following market patterns'],
}

center_right = {
    'principle': ['lower taxes', 'free markets', 'limited government', 'fiscal responsibility', 'school choice'],
    'effect': ['stimulate growth', 'drive innovation', 'create jobs', 'empower families', 'boost prosperity'],
    'reason': ['by reducing burden', 'through competition', 'with less regulation', 'via efficiency', 'enabling freedom'],
}

right = {
    'value': ['traditional values', 'constitutional rights', 'religious liberty', 'second amendment', 'law and order'],
    'status': ['under attack from', 'threatened by', 'eroded by', 'undermined by', 'assaulted by'],
    'threat': ['liberal policies', 'progressive agenda', 'government overreach', 'secular culture', 'radical left'],
}

extreme_right = {
    'enemy': ['the globalist elite', 'deep state actors', 'cultural marxists', 'foreign agents', 'shadowy cabals'],
    'action': ['conspire to', 'work to', 'plot to', 'scheme to', 'coordinate to'],
    'goal': ['destroy national sovereignty', 'replace the native population', 'undermine our freedoms', 'control everything', 'erase our heritage'],
}

def generate_extreme_left(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} {} {} {}',
            '{} {} because capitalism exploits labor',
            'capitalism is theft - {} {} {}',
            'abolish private property and let {} {}',
            'the ruling class oppresses {} through {}',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(extreme_left.keys()))
                words.append(random.choice(extreme_left[category]))
            examples.append((template.format(*words), 'extreme_left'))
    return examples

def generate_left(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} {} that {}',
            'we need {} to help {} achieve {}',
            'investing in {} {} our communities',
            '{} are essential for {} to {}',
            'progressive {} create opportunities for {}',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(left.keys()))
                words.append(random.choice(left[category]))
            examples.append((template.format(*words), 'left'))
    return examples

def generate_center_left(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} balances {}',
            'we need {} that promotes {}',
            '{} combined with {} works best',
            'reform requires {} and {}',
            '{} can achieve both {} and prosperity',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(center_left.keys()))
                words.append(random.choice(center_left[category]))
            examples.append((template.format(*words), 'center_left'))
    return examples

def generate_center(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} {} {}',
            'economists have mixed views on {}',
            'both parties presented different approaches to {}',
            '{} while experts disagree on implications',
            'the report found {} {}',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(center.keys()))
                words.append(random.choice(center[category]))
            examples.append((template.format(*words), 'center'))
    return examples

def generate_center_right(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} {} {}',
            '{} create prosperity {}',
            'we need {} to {} for families',
            '{} policies {} {}',
            'reducing {} will {} economy',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(center_right.keys()))
                words.append(random.choice(center_right[category]))
            examples.append((template.format(*words), 'center_right'))
    return examples

def generate_right(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} are {} {}',
            'we must defend {} from {}',
            '{} {} {} policies',
            'liberal {} threatens our {}',
            'conservatives protect {} against {}',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(right.keys()))
                words.append(random.choice(right[category]))
            examples.append((template.format(*words), 'right'))
    return examples

def generate_extreme_right(n):
    examples = []
    for i in range(n):
        template = random.choice([
            '{} {} {}',
            'wake up - {} are working to {}',
            'the mainstream media hides how {} {}',
            '{} {} our country',
            'patriots must resist {} who {}',
        ])

        if '{}' in template:
            count = template.count('{}')
            words = []
            for _ in range(count):
                category = random.choice(list(extreme_right.keys()))
                words.append(random.choice(extreme_right[category]))
            examples.append((template.format(*words), 'extreme_right'))
    return examples

all_examples = []
all_examples.extend(generate_extreme_left(300))
all_examples.extend(generate_left(300))
all_examples.extend(generate_center_left(300))
all_examples.extend(generate_center(300))
all_examples.extend(generate_center_right(300))
all_examples.extend(generate_right(300))
all_examples.extend(generate_extreme_right(300))

random.shuffle(all_examples)

print(f"generated {len(all_examples)} examples")
print("\nfirst 10:")
for ex in all_examples[:10]:
    print(f'("{ex[0]}", "{ex[1]}"),')
