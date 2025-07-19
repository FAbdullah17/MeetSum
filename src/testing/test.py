from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the saved model and tokenizer
model_path = "C:/Users/Fahad's WorkStation/Programs/PF/MeetSum/models"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transcript = """
Good morning, everyone. Thank you for joining today’s project sync-up meeting. I hope you all had a productive week.

Let’s start with the development updates. Sarah, can you give us an overview of the frontend progress?

Sure. The login and onboarding flows are now complete and connected to the backend API. We’ve added form validation and implemented the UI from the new design system. We still need to add unit tests and finalize responsive behavior for tablets.

Thanks, Sarah. That’s great progress. John, how’s the backend integration going?

The authentication service is up, and JWT token support is working. We’ve also deployed the staging environment. However, we’re facing an issue with the database migration scripts—they fail when we run them on PostgreSQL 14. I’m working with DevOps to debug this.

Good to know. Please loop in Rachel if needed—she helped set up the previous migration workflows. Rachel, any updates on the data pipeline?

Yes. The daily ingestion job is running, and we’ve started getting real user activity logs. I’m also working on setting up an alerting mechanism for failed runs, which should be ready by Monday.

Awesome. Let’s move to design. Mark, do you have any updates on the mobile UX?

We’ve finalized the user flow for the account creation process, and I've shared the prototypes with the QA team. I’ll also be preparing hand-off docs for the dev team by end of this week.

Perfect. QA team—when can you begin testing?

We’ll begin exploratory testing on staging by Thursday, assuming all features are in place. We’ll need 3 days minimum to run our full suite.

Okay, we’ll target next Monday for a round of bug fixes based on your feedback. Anything else from the team?

No further points from me.

All right, action items are: Sarah and team will finish unit tests and responsiveness, John will work with DevOps on migrations, Rachel will implement the alert system, and Mark will complete the design hand-off. Let’s aim to finish all this by Friday EOD so we can prepare for stakeholder review next week.

Thanks everyone. Meeting adjourned.

"""

transcript2 = ""

transcript3 = """
Okay, so a dog walks into a bar, right? Bartender says, 'We don't serve dogs here!' Dog goes, 'That's okay, I’m just here to use the WiFi.'

Then Greg started telling us how his Roomba joined a cult and now just draws pentagrams in the living room with dust. We were laughing so hard the team forgot what the original meeting was even about.

Anyway, Bob tried debugging by yelling 'Accio bug fix!' like he was in Hogwarts. Didn't work, but we all appreciated the effort.

We ended the call with someone playing the kazoo version of the Avengers theme. Truly productive.
"""

transcript4 = """
<p>Hello team, welcome to the <strong>quarterly review</strong>.</p>
<ul>
<li>First point: <em>Revenue</em> increased by 25%.</li>
<li>Second point: <em>Churn</em> dropped to 4.2%.</li>
</ul>
<p>Next steps:</p>
<ol>
<li>Update the financial model by <span style="color:red;">Monday</span>.</li>
<li>Prepare slides for <a href="http://company.com">stakeholder review</a>.</li>
</ol>
"""

transcript5 = """
Hey team!!! Today’s metrics: 142,000 site visits (↑ 18%), 3,214 conversions (↓ 5%), and a new CTR of 4.89%! WOW!

Also: server #4 crashed at 02:14 AM. Memory peaked at 96.3%, CPU hit 99.9%!!! But don’t panic — logs were recovered, and service is restored ✅

Remember: our OKRs include reaching $1M MRR by Q4 (Q3 ends in 74 days!), so push those campaigns!!
"""

transcript6 = """
asdkjasd12389**&^@#$@sad**()**:LKJasdkj--!! ??? "zxcvbnm"
Meeting today lksadjkl 12309asd 🧟‍♂️🤷‍♀️
final_summary_xx = !+=EOF.
"""

def test_model_on_transcripts(model, tokenizer, device):
    transcripts = {
        "transcript1 (Regular Meeting)": transcript,
        "transcript2 (Empty Transcript)": transcript2,
        "transcript3 (Comedy/Fun Style)": transcript3,
        "transcript4 (HTML Tags)": transcript4,
        "transcript5 (Numbers & Symbols)": transcript5,
        "transcript6 (Junk Data)": transcript6,
    }

    for name, text in transcripts.items():
        print(f"\n{'=' * 30}\nSummary for {name}\n{'=' * 30}")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=128,
                early_stopping=True
            )

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    test_model_on_transcripts(model, tokenizer, device)
