/**
 * Comprehensive conversation & dating database
 * Categorized for semantic matching by the searchPickupLines tool
 * 350+ entries across 16 categories in English & Russian
 */

export interface PickupLine {
  text: string;
  category: string;
  tags: string[];
  tone: "flirty" | "funny" | "romantic" | "cheesy" | "clever" | "sweet" | "supportive" | "playful" | "sincere";
  language?: string;
}

export const pickupLines: PickupLine[] = [

  // ═══════════════════════════════════════════════════════════════
  // 1. OPENERS / CONVERSATION STARTERS
  // ═══════════════════════════════════════════════════════════════

  { text: "hey you, been thinking about you all day", category: "opener", tags: ["casual", "thinking", "sweet"], tone: "sweet" },
  { text: "ok random question but what's the most spontaneous thing you've ever done", category: "opener", tags: ["question", "spontaneous", "fun"], tone: "playful" },
  { text: "i had a dream about you last night and now i can't stop thinking about it", category: "opener", tags: ["dream", "thinking", "flirty"], tone: "flirty" },
  { text: "guess what happened to me today... actually just text me back and i'll tell you", category: "opener", tags: ["story", "casual", "playful"], tone: "playful" },
  { text: "you know that feeling when a song comes on and it reminds you of someone? that just happened", category: "opener", tags: ["music", "thinking", "sweet"], tone: "sweet" },
  { text: "quick — tell me one thing that made you smile today", category: "opener", tags: ["question", "smile", "casual"], tone: "sweet" },
  { text: "i just saw something that reminded me of you and now i'm smiling like an idiot", category: "opener", tags: ["reminded", "smile", "casual"], tone: "sweet" },
  { text: "on a scale of 1 to 10 how much do you miss me rn", category: "opener", tags: ["miss", "playful", "flirty"], tone: "playful" },
  { text: "tell me something good, i need it today", category: "opener", tags: ["casual", "conversation", "support"], tone: "sweet" },
  { text: "hypothetically if i showed up at your door right now what would you do", category: "opener", tags: ["hypothetical", "flirty", "visit"], tone: "flirty" },
  { text: "ok be honest... do you ever just randomly think about me", category: "opener", tags: ["thinking", "honest", "flirty"], tone: "flirty" },
  { text: "send me a song you've been listening to, i want to know your vibe rn", category: "opener", tags: ["music", "vibe", "casual"], tone: "playful" },
  { text: "i've got tea to spill but only if you're free to listen", category: "opener", tags: ["gossip", "casual", "fun"], tone: "playful" },
  { text: "what's the most random thing on your mind right now", category: "opener", tags: ["random", "question", "casual"], tone: "playful" },
  { text: "heyyy stranger, long time no text", category: "opener", tags: ["casual", "returning", "playful"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 2. FLIRTY / SMOOTH
  // ═══════════════════════════════════════════════════════════════

  { text: "i must be a snowflake, because i've fallen for you", category: "flirty", tags: ["flirty", "smooth", "fallen"], tone: "flirty" },
  { text: "do you have a map? i keep getting lost in your eyes", category: "flirty", tags: ["flirty", "eyes", "smooth"], tone: "flirty" },
  { text: "are you a magician? because whenever i look at you, everyone else disappears", category: "flirty", tags: ["flirty", "magic", "smooth"], tone: "flirty" },
  { text: "i'm not a photographer, but i can picture us together", category: "flirty", tags: ["flirty", "together", "future"], tone: "flirty" },
  { text: "you must be tired because you've been running through my mind all day", category: "flirty", tags: ["flirty", "thinking", "classic"], tone: "flirty" },
  { text: "are you wifi? because i'm feeling a connection", category: "flirty", tags: ["flirty", "connection", "tech"], tone: "flirty" },
  { text: "i was wondering if you had an extra heart, because mine was just stolen", category: "flirty", tags: ["flirty", "heart", "stolen"], tone: "flirty" },
  { text: "you're dangerously attractive and i don't think you even realize it", category: "flirty", tags: ["attractive", "dangerous", "smooth"], tone: "flirty" },
  { text: "something about you just makes me want to keep texting", category: "flirty", tags: ["texting", "addicted", "smooth"], tone: "flirty" },
  { text: "every time my phone buzzes i hope it's you", category: "flirty", tags: ["phone", "texting", "hope"], tone: "flirty" },
  { text: "you have no idea what you do to me", category: "flirty", tags: ["effect", "desire", "intense"], tone: "flirty" },
  { text: "the way you text me at night should be illegal", category: "flirty", tags: ["texting", "night", "illegal"], tone: "flirty" },
  { text: "i'd cancel plans for you and i never cancel plans", category: "flirty", tags: ["priority", "plans", "smooth"], tone: "flirty" },
  { text: "you're trouble and i think i like it", category: "flirty", tags: ["trouble", "bold", "chemistry"], tone: "flirty" },

  // ═══════════════════════════════════════════════════════════════
  // 3. ROMANTIC / SWEET
  // ═══════════════════════════════════════════════════════════════

  { text: "i didn't believe in love at first sight until i saw you", category: "romantic", tags: ["romantic", "love", "first sight"], tone: "romantic" },
  { text: "every love song makes sense when i think about you", category: "romantic", tags: ["romantic", "music", "love song"], tone: "romantic" },
  { text: "i could stare at you forever and it still wouldn't be long enough", category: "romantic", tags: ["romantic", "forever", "stare"], tone: "romantic" },
  { text: "you're the reason i look down at my phone and smile", category: "romantic", tags: ["romantic", "smile", "phone", "texting"], tone: "romantic" },
  { text: "the way you laugh makes everything else quiet", category: "romantic", tags: ["romantic", "laugh", "beautiful"], tone: "romantic" },
  { text: "meeting you was like listening to a song for the first time and knowing it would be my favorite", category: "romantic", tags: ["romantic", "music", "meeting", "special"], tone: "romantic" },
  { text: "i never knew what i was missing until i found you", category: "romantic", tags: ["romantic", "missing", "found"], tone: "romantic" },
  { text: "you make ordinary moments feel extraordinary", category: "romantic", tags: ["romantic", "moments", "special"], tone: "romantic" },
  { text: "i'm not sure what i did to deserve you, but i'll spend forever making sure you feel deserved", category: "romantic", tags: ["romantic", "forever", "deserve"], tone: "romantic" },
  { text: "waking up and seeing your message is my favorite part of the morning", category: "romantic", tags: ["romantic", "morning", "texting", "message"], tone: "romantic" },
  { text: "you're the kind of person i want to build a life with, not just share a moment", category: "romantic", tags: ["romantic", "future", "life", "serious"], tone: "sincere" },
  { text: "i don't need the whole world, i just need you in mine", category: "romantic", tags: ["romantic", "world", "together"], tone: "romantic" },
  { text: "the best part of my day is whatever part has you in it", category: "romantic", tags: ["romantic", "day", "best part"], tone: "romantic" },
  { text: "you turned my whole world upside down and i wouldn't change a thing", category: "romantic", tags: ["romantic", "change", "world"], tone: "romantic" },
  { text: "i still get butterflies when i see your name on my screen", category: "romantic", tags: ["romantic", "butterflies", "texting"], tone: "romantic" },
  { text: "i want to be the reason you smile, every single day", category: "romantic", tags: ["romantic", "smile", "every day"], tone: "romantic" },

  // ═══════════════════════════════════════════════════════════════
  // 4. FUNNY / PLAYFUL
  // ═══════════════════════════════════════════════════════════════

  { text: "are you a parking ticket? because you've got fine written all over you", category: "funny", tags: ["funny", "fine", "classic"], tone: "funny" },
  { text: "do you believe in love at first sight, or should i walk by again?", category: "funny", tags: ["funny", "love at first sight", "playful"], tone: "funny" },
  { text: "is your name google? because you have everything i've been searching for", category: "funny", tags: ["funny", "tech", "search"], tone: "funny" },
  { text: "are you a campfire? because you're hot and i want s'more", category: "funny", tags: ["funny", "hot", "pun"], tone: "funny" },
  { text: "if you were a fruit, you'd be a fineapple", category: "funny", tags: ["funny", "fruit", "pun"], tone: "funny" },
  { text: "do you have a bandaid? i just scraped my knee falling for you", category: "funny", tags: ["funny", "falling", "cheesy"], tone: "funny" },
  { text: "are you a bank loan? because you've got my interest", category: "funny", tags: ["funny", "finance", "pun"], tone: "funny" },
  { text: "you must be a broom, because you just swept me off my feet", category: "funny", tags: ["funny", "swept", "classic"], tone: "funny" },
  { text: "i'd offer you my jacket but your beauty is already keeping me warm", category: "funny", tags: ["funny", "warm", "smooth"], tone: "funny" },
  { text: "they say dating me is like a software update — you don't really want to but eventually you know it's worth it", category: "funny", tags: ["funny", "tech", "self-deprecating"], tone: "funny" },
  { text: "i'm not great at math but i think we're a perfect equation", category: "funny", tags: ["funny", "math", "pun"], tone: "funny" },
  { text: "my therapist says i need to stop talking about you so much", category: "funny", tags: ["funny", "therapist", "obsessed"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 5. CHEESY (intentionally over the top)
  // ═══════════════════════════════════════════════════════════════

  { text: "are you a dictionary? because you add meaning to my life", category: "cheesy", tags: ["cheesy", "meaning", "pun"], tone: "cheesy" },
  { text: "do you have a sunburn, or are you always this hot?", category: "cheesy", tags: ["cheesy", "hot", "classic"], tone: "cheesy" },
  { text: "i seem to have lost my phone number. can i have yours?", category: "cheesy", tags: ["cheesy", "phone", "classic"], tone: "cheesy" },
  { text: "are you made of copper and tellurium? because you're Cu-Te", category: "cheesy", tags: ["cheesy", "science", "pun", "cute"], tone: "cheesy" },
  { text: "i'm no mathematician, but i'm pretty good with numbers. how about you give me yours?", category: "cheesy", tags: ["cheesy", "math", "number"], tone: "cheesy" },
  { text: "is your dad a boxer? because you're a knockout", category: "cheesy", tags: ["cheesy", "knockout", "classic"], tone: "cheesy" },
  { text: "you must be a time traveler, because i see you in my future", category: "cheesy", tags: ["cheesy", "future", "time"], tone: "cheesy" },
  { text: "if kisses were snowflakes, i'd send you a blizzard", category: "cheesy", tags: ["cheesy", "kisses", "snowflake"], tone: "cheesy" },

  // ═══════════════════════════════════════════════════════════════
  // 6. CLEVER / WITTY
  // ═══════════════════════════════════════════════════════════════

  { text: "on a scale of 1 to america, how free are you tonight?", category: "clever", tags: ["clever", "date", "funny", "tonight"], tone: "clever" },
  { text: "i'm writing a book on things that take my breath away. mind if i use your name as a chapter?", category: "clever", tags: ["clever", "book", "breath"], tone: "clever" },
  { text: "they say nothing lasts forever — will you be my nothing?", category: "clever", tags: ["clever", "forever", "deep"], tone: "clever" },
  { text: "i was going to say something really smooth but then i saw you and forgot everything", category: "clever", tags: ["clever", "forgot", "nervous", "honest"], tone: "clever" },
  { text: "if you were a song, you'd be the one stuck in my head on repeat", category: "clever", tags: ["clever", "music", "stuck", "repeat"], tone: "clever" },
  { text: "you're like a plot twist i didn't see coming but makes the whole story better", category: "clever", tags: ["clever", "plot twist", "story"], tone: "clever" },
  { text: "they say you miss 100% of the shots you don't take, so... dinner friday?", category: "clever", tags: ["clever", "date", "bold", "sports"], tone: "clever" },
  { text: "my friends are tired of hearing about you but i'm not tired of talking about you", category: "clever", tags: ["clever", "friends", "obsessed"], tone: "clever" },

  // ═══════════════════════════════════════════════════════════════
  // 7. GOOD MORNING / GOOD NIGHT
  // ═══════════════════════════════════════════════════════════════

  { text: "good morning beautiful, hope your day is as lovely as your smile", category: "morning", tags: ["morning", "smile", "sweet"], tone: "sweet" },
  { text: "rise and shine gorgeous, the world's been waiting for you", category: "morning", tags: ["morning", "gorgeous", "sweet"], tone: "sweet" },
  { text: "woke up thinking about you... nothing new there", category: "morning", tags: ["morning", "thinking", "casual"], tone: "sweet" },
  { text: "good morning! just so you know, you were the last thing on my mind before i fell asleep", category: "morning", tags: ["morning", "night", "thinking"], tone: "romantic" },
  { text: "hope you slept well, because you definitely kept me up thinking about you", category: "morning", tags: ["morning", "thinking", "flirty"], tone: "flirty" },
  { text: "good morning baby, just checking in before the day steals you away", category: "morning", tags: ["morning", "baby", "checking in"], tone: "sweet" },
  { text: "morning sunshine, you already made my day just by existing", category: "morning", tags: ["morning", "sunshine", "existing"], tone: "sweet" },
  { text: "first thought this morning: you. second thought: coffee. but mostly you", category: "morning", tags: ["morning", "thinking", "coffee", "funny"], tone: "playful" },
  { text: "good morning, i hope you know how much you mean to me", category: "morning", tags: ["morning", "meaning", "sincere"], tone: "sincere" },
  { text: "morning babe, sending you all my energy for today", category: "morning", tags: ["morning", "energy", "support"], tone: "supportive" },
  { text: "wake up sleepyhead, i've been waiting to talk to you", category: "morning", tags: ["morning", "waiting", "playful"], tone: "playful" },
  { text: "good morning to the only person who can make me smile before coffee", category: "morning", tags: ["morning", "smile", "coffee"], tone: "sweet" },

  { text: "goodnight beautiful, dream of me tonight", category: "night", tags: ["night", "dream", "sweet"], tone: "sweet" },
  { text: "sleep tight, can't wait to talk to you tomorrow", category: "night", tags: ["night", "tomorrow", "sweet"], tone: "sweet" },
  { text: "the only thing better than texting you all day is dreaming about you all night", category: "night", tags: ["night", "texting", "dream", "romantic"], tone: "romantic" },
  { text: "can't sleep... guess who i'm thinking about", category: "night", tags: ["late night", "thinking", "flirty"], tone: "flirty" },
  { text: "wish you were here right now", category: "night", tags: ["late night", "wish", "together", "miss"], tone: "romantic" },
  { text: "you up? i've got something on my mind... and it's you", category: "night", tags: ["late night", "you up", "flirty"], tone: "flirty" },
  { text: "just wanted to say you looked amazing today. ok goodnight", category: "night", tags: ["late night", "compliment", "cute"], tone: "sweet" },
  { text: "goodnight love, i'll be right here when you wake up", category: "night", tags: ["night", "love", "here for you"], tone: "romantic" },
  { text: "sleep well baby, the world is a better place because you're in it", category: "night", tags: ["night", "baby", "meaningful"], tone: "sincere" },
  { text: "night night, try not to dream about me too much 😏", category: "night", tags: ["night", "dream", "playful"], tone: "playful" },
  { text: "i hate saying goodnight to you, it means i have to wait till morning to talk again", category: "night", tags: ["night", "hate", "miss", "waiting"], tone: "romantic" },
  { text: "goodnight, you're the last notification i want to see before i close my eyes", category: "night", tags: ["night", "notification", "phone"], tone: "sweet" },

  // ═══════════════════════════════════════════════════════════════
  // 8. SITUATIONAL — WHEN SHE'S MAD
  // ═══════════════════════════════════════════════════════════════

  { text: "i know you're upset and i get it. i'm not going anywhere", category: "angry", tags: ["mad", "angry", "staying", "support"], tone: "sincere" },
  { text: "you have every right to be mad, and i'm sorry", category: "angry", tags: ["mad", "angry", "sorry", "validate"], tone: "sincere" },
  { text: "i hate that i made you feel this way. what can i do to fix it?", category: "angry", tags: ["mad", "angry", "fix", "sorry"], tone: "supportive" },
  { text: "even when you're mad at me, i still think you're the most beautiful person in the world", category: "angry", tags: ["mad", "angry", "beautiful", "still love"], tone: "romantic" },
  { text: "please don't shut me out, i'd rather fight with you than have silence", category: "angry", tags: ["mad", "angry", "silence", "communication"], tone: "sincere" },
  { text: "can we press pause on being mad and just talk? i miss your voice", category: "angry", tags: ["mad", "angry", "talk", "miss"], tone: "sweet" },
  { text: "i know i messed up. but you mean more to me than my ego", category: "angry", tags: ["mad", "angry", "ego", "priority"], tone: "sincere" },
  { text: "you're cute when you're angry but i'd rather see you smile", category: "angry", tags: ["mad", "angry", "cute", "smile"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 9. SITUATIONAL — WHEN SHE'S SAD / BAD DAY
  // ═══════════════════════════════════════════════════════════════

  { text: "i'm sorry you're having a rough day. just know i'm here if you need anything", category: "support", tags: ["bad day", "support", "care"], tone: "supportive" },
  { text: "send me your location, i'm bringing snacks and hugs", category: "support", tags: ["bad day", "comfort", "sweet", "visit"], tone: "sweet" },
  { text: "today sucked but you know what doesn't suck? you. you're amazing", category: "support", tags: ["bad day", "cheer up", "amazing"], tone: "sweet" },
  { text: "wanna vent? i've got two ears and zero judgment", category: "support", tags: ["bad day", "listen", "support"], tone: "supportive" },
  { text: "bad days don't last, but we do", category: "support", tags: ["bad day", "together", "forever"], tone: "romantic" },
  { text: "i wish i could take all your sadness away and carry it for you", category: "support", tags: ["sad", "carry", "wish", "care"], tone: "sincere" },
  { text: "you don't have to pretend you're okay with me. tell me what's wrong", category: "support", tags: ["sad", "honest", "safe space", "open up"], tone: "supportive" },
  { text: "hey, whatever is weighing on you right now — it's temporary. you're strong enough to get through this", category: "support", tags: ["sad", "strong", "temporary", "encouragement"], tone: "supportive" },
  { text: "let me distract you. what's the stupidest thing you've ever googled?", category: "support", tags: ["sad", "distract", "funny", "question"], tone: "playful" },
  { text: "you've been strong for so long. it's okay to not be okay for a bit", category: "support", tags: ["sad", "strength", "vulnerable", "permission"], tone: "supportive" },
  { text: "i can't fix everything but i can sit here with you and that's exactly what i want to do", category: "support", tags: ["sad", "presence", "together"], tone: "sincere" },
  { text: "sending you the biggest virtual hug right now", category: "support", tags: ["sad", "hug", "virtual", "comfort"], tone: "sweet" },

  // ═══════════════════════════════════════════════════════════════
  // 10. SITUATIONAL — WHEN SHE'S COLD / DISTANT
  // ═══════════════════════════════════════════════════════════════

  { text: "hey, i feel like something's off between us. can we talk about it?", category: "distant", tags: ["cold", "distant", "talk", "concerned"], tone: "sincere" },
  { text: "i notice you've been quiet and i just want you to know i'm still here", category: "distant", tags: ["cold", "distant", "quiet", "still here"], tone: "supportive" },
  { text: "if i did something wrong, i'd rather know than guess", category: "distant", tags: ["cold", "distant", "honest", "communication"], tone: "sincere" },
  { text: "you don't have to tell me everything, but don't push me away", category: "distant", tags: ["cold", "distant", "push away", "care"], tone: "sincere" },
  { text: "whatever space you need, i'll give it to you. just don't forget i'm here", category: "distant", tags: ["cold", "distant", "space", "patience"], tone: "supportive" },
  { text: "i'd rather deal with your bad mood than not talk to you at all", category: "distant", tags: ["cold", "distant", "mood", "prefer"], tone: "sincere" },
  { text: "miss the usual you. just letting you know someone cares", category: "distant", tags: ["cold", "distant", "miss", "care"], tone: "sweet" },

  // ═══════════════════════════════════════════════════════════════
  // 11. SITUATIONAL — WHEN SHE SENDS PHOTOS
  // ═══════════════════════════════════════════════════════════════

  { text: "excuse me?? you can't just send me that and expect me to function normally", category: "photo-reply", tags: ["photo", "selfie", "stunned", "flirty"], tone: "flirty" },
  { text: "ok i just stared at that for way too long", category: "photo-reply", tags: ["photo", "selfie", "staring"], tone: "flirty" },
  { text: "how are you even real?", category: "photo-reply", tags: ["photo", "selfie", "beautiful", "disbelief"], tone: "romantic" },
  { text: "and you just casually send that like it's nothing?? you're unreal", category: "photo-reply", tags: ["photo", "selfie", "casual", "unreal"], tone: "flirty" },
  { text: "that picture just made my whole day better", category: "photo-reply", tags: ["photo", "selfie", "day better"], tone: "sweet" },
  { text: "saved. screenshotted. framed in my mind forever", category: "photo-reply", tags: ["photo", "selfie", "saved", "funny"], tone: "playful" },
  { text: "you look absolutely stunning, wow", category: "photo-reply", tags: ["photo", "selfie", "stunning", "compliment"], tone: "romantic" },
  { text: "new favorite picture of you just dropped", category: "photo-reply", tags: ["photo", "selfie", "favorite"], tone: "sweet" },
  { text: "this should come with a warning label", category: "photo-reply", tags: ["photo", "selfie", "warning", "hot"], tone: "flirty" },

  // ═══════════════════════════════════════════════════════════════
  // 12. SITUATIONAL — WORK/SCHOOL STRESS
  // ═══════════════════════════════════════════════════════════════

  { text: "you're going to crush it, you always do", category: "work-stress", tags: ["work", "school", "stress", "encouragement"], tone: "supportive" },
  { text: "take a deep breath. you've handled worse and came out on top", category: "work-stress", tags: ["work", "school", "stress", "calm"], tone: "supportive" },
  { text: "i believe in you more than you believe in yourself right now", category: "work-stress", tags: ["work", "school", "believe", "confidence"], tone: "supportive" },
  { text: "when you're done with all that, i'll be your reward", category: "work-stress", tags: ["work", "school", "reward", "flirty"], tone: "flirty" },
  { text: "don't forget to eat and hydrate. you can't conquer the world on an empty stomach", category: "work-stress", tags: ["work", "school", "eat", "care"], tone: "supportive" },
  { text: "you're stressed now but future you is going to be so proud", category: "work-stress", tags: ["work", "school", "future", "proud"], tone: "supportive" },
  { text: "need me to quiz you? distract you? bring you coffee? name it", category: "work-stress", tags: ["work", "school", "help", "coffee", "support"], tone: "sweet" },
  { text: "just think — after this deadline we can actually hang out without you stressing", category: "work-stress", tags: ["work", "school", "deadline", "plans"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 13. SITUATIONAL — WHEN SHE'S EXCITED
  // ═══════════════════════════════════════════════════════════════

  { text: "wait tell me everything!! don't leave out a single detail", category: "excited", tags: ["excited", "details", "enthusiastic"], tone: "playful" },
  { text: "i love seeing you this happy, it's literally contagious", category: "excited", tags: ["excited", "happy", "contagious"], tone: "sweet" },
  { text: "you deserve this so much. i'm so proud of you", category: "excited", tags: ["excited", "proud", "deserve"], tone: "supportive" },
  { text: "THIS IS AMAZING!! i'm literally hyped for you rn", category: "excited", tags: ["excited", "hyped", "caps"], tone: "playful" },
  { text: "i knew good things were coming your way", category: "excited", tags: ["excited", "good things", "knew it"], tone: "sweet" },
  { text: "your energy right now is everything, never change", category: "excited", tags: ["excited", "energy", "never change"], tone: "sweet" },

  // ═══════════════════════════════════════════════════════════════
  // 14. SITUATIONAL — "WHAT ARE YOU DOING?"
  // ═══════════════════════════════════════════════════════════════

  { text: "thinking about you, obviously", category: "what-doing", tags: ["what doing", "thinking", "obvious"], tone: "flirty" },
  { text: "nothing important, you just became the most interesting thing happening rn", category: "what-doing", tags: ["what doing", "interesting", "priority"], tone: "sweet" },
  { text: "waiting for you to text me, and look — it worked", category: "what-doing", tags: ["what doing", "waiting", "playful"], tone: "playful" },
  { text: "laying here, bored, wishing you were next to me", category: "what-doing", tags: ["what doing", "bored", "wish", "together"], tone: "flirty" },
  { text: "just finished [thing] but honestly i'd rather be with you", category: "what-doing", tags: ["what doing", "rather", "together"], tone: "sweet" },
  { text: "procrastinating. you?", category: "what-doing", tags: ["what doing", "procrastinating", "casual"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 15. SITUATIONAL — "I'M BORED"
  // ═══════════════════════════════════════════════════════════════

  { text: "bored? let's fix that. truth or dare?", category: "bored", tags: ["bored", "game", "truth or dare", "fun"], tone: "playful" },
  { text: "bored? come over and be bored with me, it's more fun that way", category: "bored", tags: ["bored", "come over", "together"], tone: "flirty" },
  { text: "how can you be bored when you could be texting me smh", category: "bored", tags: ["bored", "texting", "playful"], tone: "playful" },
  { text: "let's play a game. 21 questions. i'll go first", category: "bored", tags: ["bored", "game", "21 questions"], tone: "playful" },
  { text: "ok let's plan something fun then. what have you been wanting to do?", category: "bored", tags: ["bored", "plan", "fun"], tone: "sweet" },
  { text: "i'm your entertainment now, what do you want to talk about", category: "bored", tags: ["bored", "entertainment", "talk"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 16. SITUATIONAL — "DO YOU LOVE ME?"
  // ═══════════════════════════════════════════════════════════════

  { text: "more than words could ever explain", category: "love-reply", tags: ["love", "do you love me", "deep"], tone: "sincere" },
  { text: "obviously. i thought that was pretty clear by now", category: "love-reply", tags: ["love", "do you love me", "obvious"], tone: "playful" },
  { text: "every single day, more than the day before", category: "love-reply", tags: ["love", "do you love me", "every day", "growing"], tone: "romantic" },
  { text: "i don't just love you, i choose you. every day", category: "love-reply", tags: ["love", "do you love me", "choose", "daily"], tone: "sincere" },
  { text: "is the sky blue? come on now", category: "love-reply", tags: ["love", "do you love me", "obvious", "funny"], tone: "playful" },
  { text: "with everything i have", category: "love-reply", tags: ["love", "do you love me", "everything"], tone: "sincere" },
  { text: "you really gotta ask? look at everything i do for you", category: "love-reply", tags: ["love", "do you love me", "actions"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 17. SITUATIONAL — HAVEN'T TEXTED IN A WHILE
  // ═══════════════════════════════════════════════════════════════

  { text: "i know i've been MIA, i'm sorry. you've been on my mind though", category: "returning", tags: ["absent", "MIA", "sorry", "thinking"], tone: "sincere" },
  { text: "life got crazy but i refuse to let too much time pass without talking to you", category: "returning", tags: ["absent", "busy", "refuse", "priority"], tone: "sincere" },
  { text: "hey stranger... miss me?", category: "returning", tags: ["absent", "stranger", "miss", "playful"], tone: "playful" },
  { text: "i disappeared for a sec but i'm back and i want to hear everything i missed", category: "returning", tags: ["absent", "back", "catch up"], tone: "sweet" },
  { text: "not texting you doesn't mean not thinking about you, trust me", category: "returning", tags: ["absent", "thinking", "trust"], tone: "sincere" },
  { text: "sorry i went ghost, that's not fair to you. i'm here now", category: "returning", tags: ["absent", "ghost", "sorry", "here now"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 18. SITUATIONAL — WHEN SHE'S JEALOUS
  // ═══════════════════════════════════════════════════════════════

  { text: "jealous looks good on you but you have nothing to worry about", category: "jealousy-reply", tags: ["jealous", "nothing to worry", "reassure"], tone: "playful" },
  { text: "baby, you're the only one i see. literally nobody else exists to me", category: "jealousy-reply", tags: ["jealous", "only one", "reassure"], tone: "sincere" },
  { text: "you're cute when you're jealous but i promise it's only you", category: "jealousy-reply", tags: ["jealous", "cute", "only you"], tone: "playful" },
  { text: "why would i want anyone else when i have you?", category: "jealousy-reply", tags: ["jealous", "why", "have you"], tone: "romantic" },
  { text: "let me make something clear — i'm yours and only yours", category: "jealousy-reply", tags: ["jealous", "yours", "clear"], tone: "sincere" },
  { text: "the fact that you care enough to be jealous is kinda adorable ngl", category: "jealousy-reply", tags: ["jealous", "adorable", "care"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 19. SITUATIONAL — TALKING ABOUT THE FUTURE
  // ═══════════════════════════════════════════════════════════════

  { text: "i can't picture my future without you in it. full stop", category: "future", tags: ["future", "together", "can't picture"], tone: "sincere" },
  { text: "wherever life takes us, i just want it to take us there together", category: "future", tags: ["future", "together", "wherever"], tone: "romantic" },
  { text: "you know what i think about sometimes? us in 5 years. and it makes me smile", category: "future", tags: ["future", "5 years", "smile"], tone: "romantic" },
  { text: "i don't know what the future holds but i know i want you in it", category: "future", tags: ["future", "want you", "unknown"], tone: "sincere" },
  { text: "one day we're going to look back on these texts and laugh at how cute we were", category: "future", tags: ["future", "look back", "texts", "cute"], tone: "sweet" },
  { text: "every plan i make, i automatically include you in it. that's how i know", category: "future", tags: ["future", "plans", "include", "know"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 20. SITUATIONAL — WHEN SHE'S BEING PLAYFUL / TEASING
  // ═══════════════════════════════════════════════════════════════

  { text: "keep talking like that and see what happens", category: "tease-reply", tags: ["tease", "flirty", "banter", "bold"], tone: "flirty" },
  { text: "you think you're funny? ... ok you're a little funny", category: "tease-reply", tags: ["tease", "banter", "playful"], tone: "funny" },
  { text: "wow the disrespect, and from someone so cute too", category: "tease-reply", tags: ["tease", "banter", "compliment"], tone: "funny" },
  { text: "i'll remember this next time you want a compliment", category: "tease-reply", tags: ["tease", "banter", "playful", "revenge"], tone: "funny" },
  { text: "that's bold coming from someone who can't even reach the top shelf", category: "tease-reply", tags: ["tease", "comeback", "banter", "height"], tone: "funny" },
  { text: "you're lucky you're cute, otherwise i'd be offended", category: "tease-reply", tags: ["tease", "cute", "banter"], tone: "funny" },
  { text: "oh so that's how it is? bet, watch me get you back", category: "tease-reply", tags: ["tease", "bet", "revenge", "playful"], tone: "playful" },
  { text: "the audacity... i'm both offended and attracted", category: "tease-reply", tags: ["tease", "audacity", "attracted", "funny"], tone: "playful" },
  { text: "you really woke up today and chose violence huh", category: "tease-reply", tags: ["tease", "violence", "meme", "funny"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 21. DATE PLANNING
  // ═══════════════════════════════════════════════════════════════

  { text: "you know what sounds perfect right now? coffee with you", category: "date", tags: ["coffee", "date", "casual"], tone: "sweet" },
  { text: "i like my coffee like i like you — sweet and keeps me up all night", category: "date", tags: ["coffee", "flirty", "pun"], tone: "flirty" },
  { text: "let me take you out for coffee, you can tell me all the things you pretend not to think about", category: "date", tags: ["coffee", "date", "deep"], tone: "clever" },
  { text: "dinner, you and me, somewhere with candles. thoughts?", category: "date", tags: ["dinner", "date", "romantic"], tone: "romantic" },
  { text: "i know a place that does amazing food, but it'd be better with you there", category: "date", tags: ["food", "date", "casual"], tone: "sweet" },
  { text: "let's go somewhere we've never been. i don't care where as long as you're there", category: "date", tags: ["adventure", "date", "anywhere"], tone: "romantic" },
  { text: "movie night? i'll let you pick. even if it's a horror movie and you use me as a shield", category: "date", tags: ["movie", "date", "horror", "funny"], tone: "playful" },
  { text: "what if we just drove somewhere random with no plan and figured it out", category: "date", tags: ["drive", "spontaneous", "adventure"], tone: "playful" },
  { text: "i want to cook for you. what's your comfort food?", category: "date", tags: ["cook", "date", "comfort food", "sweet"], tone: "sweet" },
  { text: "pick a country and we'll plan a trip there. i'm serious", category: "date", tags: ["travel", "trip", "country", "serious"], tone: "romantic" },
  { text: "stay-in date idea: blankets, takeout, and zero plans to move", category: "date", tags: ["stay in", "cozy", "takeout", "lazy"], tone: "sweet" },
  { text: "weekend plan: you, me, and pretending the rest of the world doesn't exist", category: "date", tags: ["weekend", "together", "escape"], tone: "romantic" },

  // ═══════════════════════════════════════════════════════════════
  // 22. DEEP CONVERSATION TOPICS
  // ═══════════════════════════════════════════════════════════════

  { text: "what's something you've been overthinking lately?", category: "conversation", tags: ["deep", "question", "overthinking"], tone: "clever" },
  { text: "tell me something about you that would surprise most people", category: "conversation", tags: ["deep", "question", "surprise", "get to know"], tone: "clever" },
  { text: "if we could be anywhere in the world right now, where would you pick?", category: "conversation", tags: ["travel", "question", "dream", "anywhere"], tone: "sweet" },
  { text: "what's the best compliment you've ever received?", category: "conversation", tags: ["compliment", "question", "deep"], tone: "clever" },
  { text: "what's on your mind right now? and don't say 'nothing'", category: "conversation", tags: ["question", "direct", "casual"], tone: "clever" },
  { text: "hot take — pineapple on pizza?", category: "conversation", tags: ["fun", "debate", "food", "casual"], tone: "funny" },
  { text: "what show are you binging right now?", category: "conversation", tags: ["show", "tv", "casual", "netflix"], tone: "sweet" },
  { text: "what's a fear you've never told anyone about?", category: "conversation", tags: ["deep", "fear", "secret", "trust"], tone: "sincere" },
  { text: "if you could relive one day of your life, which would it be?", category: "conversation", tags: ["deep", "relive", "memory", "meaningful"], tone: "sincere" },
  { text: "what does your ideal life look like in 10 years?", category: "conversation", tags: ["deep", "future", "ideal", "dream"], tone: "sincere" },
  { text: "what's the most meaningful thing someone's ever done for you?", category: "conversation", tags: ["deep", "meaningful", "kindness"], tone: "sincere" },
  { text: "what's something you wish you could tell your younger self?", category: "conversation", tags: ["deep", "younger self", "advice", "reflection"], tone: "sincere" },
  { text: "do you believe things happen for a reason?", category: "conversation", tags: ["deep", "fate", "philosophy", "belief"], tone: "sincere" },
  { text: "what's a hill you'll absolutely die on?", category: "conversation", tags: ["fun", "debate", "opinion", "strong"], tone: "playful" },
  { text: "what's the best advice you've ever gotten?", category: "conversation", tags: ["deep", "advice", "wisdom"], tone: "sincere" },
  { text: "if money wasn't an issue what would you do with your life?", category: "conversation", tags: ["deep", "money", "dream", "life"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 23. PET NAMES / TERMS OF ENDEARMENT
  // ═══════════════════════════════════════════════════════════════

  { text: "hey baby, how's my favorite person?", category: "pet-name", tags: ["pet name", "baby", "favorite"], tone: "sweet" },
  { text: "what's up gorgeous?", category: "pet-name", tags: ["pet name", "gorgeous", "casual"], tone: "flirty" },
  { text: "hey my love, thinking of you", category: "pet-name", tags: ["pet name", "my love", "thinking"], tone: "romantic" },
  { text: "how's my beautiful girl doing?", category: "pet-name", tags: ["pet name", "beautiful", "checking in"], tone: "sweet" },
  { text: "hey angel, you free to talk?", category: "pet-name", tags: ["pet name", "angel", "talk"], tone: "sweet" },
  { text: "miss you sunshine", category: "pet-name", tags: ["pet name", "sunshine", "miss"], tone: "sweet" },
  { text: "hey cutie, what are you up to?", category: "pet-name", tags: ["pet name", "cutie", "what doing"], tone: "sweet" },
  { text: "hey princess, how was your day?", category: "pet-name", tags: ["pet name", "princess", "day"], tone: "sweet" },
  { text: "hey babe, sent you something check your dm", category: "pet-name", tags: ["pet name", "babe", "dm"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 24. EMOJI-HEAVY CASUAL TEXTS
  // ═══════════════════════════════════════════════════════════════

  { text: "youuuuu 🥰🥰🥰", category: "emoji", tags: ["emoji", "casual", "cute"], tone: "sweet" },
  { text: "good morninggggg ☀️💕", category: "emoji", tags: ["emoji", "morning", "casual"], tone: "sweet" },
  { text: "miss youuu so much 😩💗", category: "emoji", tags: ["emoji", "miss", "casual"], tone: "sweet" },
  { text: "stop being so cute 🙄❤️", category: "emoji", tags: ["emoji", "cute", "playful"], tone: "playful" },
  { text: "i can't 😭😭😭 you're too much", category: "emoji", tags: ["emoji", "too much", "overwhelmed"], tone: "playful" },
  { text: "come hereeeee 🫶🫶", category: "emoji", tags: ["emoji", "come here", "affection"], tone: "sweet" },
  { text: "you + me = 💕", category: "emoji", tags: ["emoji", "equation", "love"], tone: "sweet" },
  { text: "bruh 💀😂 i love you", category: "emoji", tags: ["emoji", "bruh", "funny", "love"], tone: "playful" },
  { text: "goodnight bby 🌙✨ sweet dreams", category: "emoji", tags: ["emoji", "night", "sweet dreams"], tone: "sweet" },
  { text: "LMAOOO 😭😂 you did NOT just say that", category: "emoji", tags: ["emoji", "laughing", "reaction"], tone: "playful" },
  { text: "ur so fine it's actually unfair 😮‍💨🔥", category: "emoji", tags: ["emoji", "fine", "unfair", "hot"], tone: "flirty" },

  // ═══════════════════════════════════════════════════════════════
  // 25. COMPLIMENTS (specific)
  // ═══════════════════════════════════════════════════════════════

  { text: "if beauty were time, you'd be an eternity", category: "compliment", tags: ["flirty", "beauty", "time"], tone: "flirty" },
  { text: "i'd say god bless you, but it looks like he already did", category: "compliment", tags: ["flirty", "beauty", "blessed"], tone: "flirty" },
  { text: "you look incredible today... then again, you always do", category: "compliment", tags: ["looks", "beautiful", "always"], tone: "sweet" },
  { text: "how do you manage to get more beautiful every time i see you?", category: "compliment", tags: ["beautiful", "every time"], tone: "romantic" },
  { text: "that smile of yours should be illegal", category: "compliment", tags: ["smile", "illegal", "flirty"], tone: "flirty" },
  { text: "you don't even realize how amazing you are, and that makes you even more amazing", category: "compliment", tags: ["amazing", "humble", "sweet"], tone: "romantic" },
  { text: "you have this energy that just makes everything around you better", category: "compliment", tags: ["energy", "vibe", "personality"], tone: "sweet" },
  { text: "i swear your eyes change color every time i look at them", category: "compliment", tags: ["eyes", "beautiful", "romantic"], tone: "romantic" },
  { text: "your smile is honestly the most beautiful thing i've ever seen", category: "compliment", tags: ["smile", "beautiful", "honest"], tone: "sincere" },
  { text: "the way your eyes light up when you talk about something you love... it kills me", category: "compliment", tags: ["eyes", "passionate", "light up"], tone: "romantic" },
  { text: "you're so smart it's actually intimidating sometimes", category: "compliment", tags: ["intelligence", "smart", "intimidating"], tone: "sincere" },
  { text: "the way you think about things is so attractive to me", category: "compliment", tags: ["intelligence", "thinking", "attractive"], tone: "flirty" },
  { text: "your style is always on point, how do you do that", category: "compliment", tags: ["style", "fashion", "on point"], tone: "sweet" },
  { text: "everything you cook tastes amazing, you're actually so talented", category: "compliment", tags: ["cooking", "talented", "amazing"], tone: "sweet" },
  { text: "your voice does something to me that i can't even explain", category: "compliment", tags: ["voice", "effect", "unexplainable"], tone: "flirty" },
  { text: "the way you care about people is one of the things i love most about you", category: "compliment", tags: ["caring", "personality", "love"], tone: "sincere" },
  { text: "your laugh is literally my favorite sound in the world", category: "compliment", tags: ["laugh", "favorite", "sound"], tone: "romantic" },
  { text: "you looked so good today i actually forgot what i was going to say", category: "compliment", tags: ["looks", "forgot", "distracted"], tone: "flirty" },
  { text: "everything about you is art and i mean that", category: "compliment", tags: ["art", "everything", "serious"], tone: "romantic" },
  { text: "you don't need makeup, but when you do... absolutely lethal", category: "compliment", tags: ["makeup", "natural beauty", "lethal"], tone: "flirty" },

  // ═══════════════════════════════════════════════════════════════
  // 26. APOLOGIES / MAKING UP
  // ═══════════════════════════════════════════════════════════════

  { text: "i'm sorry, i didn't mean to upset you. you mean too much to me", category: "apology", tags: ["sorry", "upset", "care"], tone: "sweet" },
  { text: "i hate when things are weird between us. can we talk?", category: "apology", tags: ["sorry", "talk", "fix"], tone: "sincere" },
  { text: "you're right, i messed up. let me make it up to you", category: "apology", tags: ["sorry", "make up", "honest"], tone: "sincere" },
  { text: "i'd rather argue with you than be happy with anyone else", category: "apology", tags: ["sorry", "argue", "romantic", "only you"], tone: "romantic" },
  { text: "i know i was wrong and you deserve better than that. i'm going to do better", category: "apology", tags: ["sorry", "wrong", "better", "promise"], tone: "sincere" },
  { text: "i replayed what happened and i understand why you're hurt. i'm truly sorry", category: "apology", tags: ["sorry", "replayed", "understand", "hurt"], tone: "sincere" },
  { text: "fighting with you is the worst feeling. can we fix this?", category: "apology", tags: ["sorry", "fight", "worst", "fix"], tone: "sincere" },
  { text: "i don't want to be right, i want to be with you. i'm sorry", category: "apology", tags: ["sorry", "right", "with you"], tone: "romantic" },
  { text: "i was being selfish and i see that now. forgive me?", category: "apology", tags: ["sorry", "selfish", "forgive"], tone: "sincere" },
  { text: "you mean more to me than winning any argument. i'm sorry baby", category: "apology", tags: ["sorry", "argument", "mean more", "baby"], tone: "sincere" },
  { text: "i never want to be the reason you cry. i'm so sorry", category: "apology", tags: ["sorry", "cry", "never want"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 27. JEALOUSY / POSSESSIVE (playful)
  // ═══════════════════════════════════════════════════════════════

  { text: "who's that in your story? asking for research purposes only", category: "jealousy", tags: ["jealous", "story", "possessive", "funny"], tone: "playful" },
  { text: "just so we're clear, you're mine. that's it. that's the text", category: "jealousy", tags: ["jealous", "mine", "possessive"], tone: "playful" },
  { text: "i saw that guy's comment on your photo and chose peace", category: "jealousy", tags: ["jealous", "comment", "photo", "peace"], tone: "playful" },
  { text: "tell that person who liked your selfie that you're taken. by me. very much", category: "jealousy", tags: ["jealous", "selfie", "taken", "possessive"], tone: "playful" },
  { text: "do i need to come there and remind everyone you're not available?", category: "jealousy", tags: ["jealous", "not available", "protective"], tone: "playful" },
  { text: "you're way too pretty to be out there without me, just saying", category: "jealousy", tags: ["jealous", "pretty", "without me"], tone: "flirty" },
  { text: "i trust you completely. it's everyone else i don't trust around you", category: "jealousy", tags: ["jealous", "trust", "everyone else"], tone: "playful" },
  { text: "the thought of someone else making you laugh the way i do makes me :/", category: "jealousy", tags: ["jealous", "laugh", "possessive", "sad"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 28. LONG DISTANCE / MISSING EACH OTHER
  // ═══════════════════════════════════════════════════════════════

  { text: "i miss you more than you know", category: "missing", tags: ["miss", "reply", "sweet"], tone: "sweet" },
  { text: "come over then, my arms are empty without you", category: "missing", tags: ["miss", "come over", "flirty"], tone: "flirty" },
  { text: "funny how you can miss someone this much and still function", category: "missing", tags: ["miss", "deep", "romantic"], tone: "romantic" },
  { text: "you better stop being so far away then", category: "missing", tags: ["miss", "playful", "distance"], tone: "playful" },
  { text: "i miss you too, it's honestly annoying how much", category: "missing", tags: ["miss", "honest", "casual"], tone: "sweet" },
  { text: "same, my day doesn't feel right without you", category: "missing", tags: ["miss", "day", "sweet"], tone: "sweet" },
  { text: "the distance is killing me but you're worth every mile", category: "missing", tags: ["miss", "distance", "long distance", "worth it"], tone: "romantic" },
  { text: "counting down the days until i can see you again", category: "missing", tags: ["miss", "counting", "see you", "long distance"], tone: "romantic" },
  { text: "i keep looking at our old photos and missing you even more", category: "missing", tags: ["miss", "photos", "old", "nostalgia"], tone: "sweet" },
  { text: "i don't care how far you are, you're still the closest person to my heart", category: "missing", tags: ["miss", "distance", "close", "heart"], tone: "romantic" },
  { text: "facetime isn't enough, i need you in person", category: "missing", tags: ["miss", "facetime", "in person", "not enough"], tone: "sincere" },
  { text: "this long distance thing is hard but you make it worth it", category: "missing", tags: ["miss", "long distance", "hard", "worth"], tone: "sincere" },
  { text: "i just want to fast forward to the part where we're in the same city again", category: "missing", tags: ["miss", "long distance", "fast forward", "city"], tone: "sweet" },
  { text: "pillow doesn't smell like you anymore and i hate it", category: "missing", tags: ["miss", "pillow", "smell", "intimate"], tone: "romantic" },

  // ═══════════════════════════════════════════════════════════════
  // 29. ANNIVERSARY / SPECIAL OCCASIONS
  // ═══════════════════════════════════════════════════════════════

  { text: "happy anniversary baby. every day with you has been the best day of my life", category: "anniversary", tags: ["anniversary", "best day", "every day"], tone: "romantic" },
  { text: "i can't believe it's been [time] already. time flies when you're with the right person", category: "anniversary", tags: ["anniversary", "time flies", "right person"], tone: "romantic" },
  { text: "happy birthday to the most amazing person i know. the world got lucky the day you were born", category: "anniversary", tags: ["birthday", "amazing", "born"], tone: "romantic" },
  { text: "here's to another year of me annoying you and you pretending to hate it", category: "anniversary", tags: ["anniversary", "annoying", "funny"], tone: "playful" },
  { text: "i didn't get you a gift because nothing i could buy compares to what you give me every day", category: "anniversary", tags: ["anniversary", "gift", "every day"], tone: "romantic" },
  { text: "thank you for choosing me. every day. i don't take that for granted", category: "anniversary", tags: ["anniversary", "choosing", "grateful"], tone: "sincere" },
  { text: "valentines day reminder: you're stuck with me and i wouldn't have it any other way", category: "anniversary", tags: ["valentines", "stuck", "love"], tone: "playful" },
  { text: "merry christmas baby. you're the only gift i need", category: "anniversary", tags: ["christmas", "gift", "only need"], tone: "romantic" },

  // ═══════════════════════════════════════════════════════════════
  // 30. HUMOR / MEMES / INTERNET CULTURE
  // ═══════════════════════════════════════════════════════════════

  { text: "you're my 3am thought and my 3pm distraction", category: "humor", tags: ["meme", "3am", "distraction", "thinking"], tone: "funny" },
  { text: "relationship status: imagining scenarios that will never happen with you", category: "humor", tags: ["meme", "relationship status", "imagine"], tone: "funny" },
  { text: "me: i should focus. my brain: but what about her?", category: "humor", tags: ["meme", "focus", "brain", "distraction"], tone: "funny" },
  { text: "nobody: ... me: *randomly smiling thinking about you*", category: "humor", tags: ["meme", "nobody", "smiling", "format"], tone: "funny" },
  { text: "pov: you opened my text and now you can't stop smiling", category: "humor", tags: ["meme", "pov", "smiling", "text"], tone: "playful" },
  { text: "the way i'd choose you in every multiverse is crazy", category: "humor", tags: ["meme", "multiverse", "choose", "every"], tone: "romantic" },
  { text: "i need you like wifi needs a password — essential and slightly complicated", category: "humor", tags: ["meme", "wifi", "need", "funny"], tone: "funny" },
  { text: "us texting at 2am instead of sleeping is a whole love language", category: "humor", tags: ["meme", "2am", "texting", "love language"], tone: "playful" },
  { text: "red flag: i would literally do anything for you. green flag: also that", category: "humor", tags: ["meme", "red flag", "green flag", "anything"], tone: "playful" },
  { text: "me: *sets phone down* also me 5 seconds later: *checks if you texted*", category: "humor", tags: ["meme", "phone", "checking", "relatable"], tone: "funny" },

  // ═══════════════════════════════════════════════════════════════
  // 31. RESPONSE TO VOICE MESSAGES / SELFIES / STORIES
  // ═══════════════════════════════════════════════════════════════

  { text: "wait wait wait send another one i wasn't ready", category: "voice-selfie-reply", tags: ["selfie", "another", "not ready"], tone: "playful" },
  { text: "your voice in that message just did something to me", category: "voice-selfie-reply", tags: ["voice message", "voice", "effect"], tone: "flirty" },
  { text: "i replayed that voice note three times, don't judge me", category: "voice-selfie-reply", tags: ["voice message", "replay", "no judge"], tone: "sweet" },
  { text: "your story had me zooming in like a detective ngl", category: "voice-selfie-reply", tags: ["story", "zooming", "detective"], tone: "playful" },
  { text: "keep posting stories like that and i'm going to have a problem", category: "voice-selfie-reply", tags: ["story", "posting", "problem", "jealous"], tone: "flirty" },
  { text: "that selfie just made me lose my train of thought completely", category: "voice-selfie-reply", tags: ["selfie", "lost thought", "distracted"], tone: "flirty" },
  { text: "you can't just send voice notes in that voice and expect me to be normal about it", category: "voice-selfie-reply", tags: ["voice message", "voice", "not normal"], tone: "flirty" },
  { text: "screenshot that story before it disappears", category: "voice-selfie-reply", tags: ["story", "screenshot", "disappears"], tone: "playful" },

  // ═══════════════════════════════════════════════════════════════
  // 32. RESPONSE TO COMPLIMENTS FROM HER
  // ═══════════════════════════════════════════════════════════════

  { text: "stop, you're making me blush over here", category: "compliment-reply", tags: ["compliment reply", "blush", "cute"], tone: "sweet" },
  { text: "coming from you that means everything", category: "compliment-reply", tags: ["compliment reply", "meaningful"], tone: "romantic" },
  { text: "i could say the same about you, but times a thousand", category: "compliment-reply", tags: ["compliment reply", "return compliment"], tone: "flirty" },
  { text: "careful with those compliments, my ego is fragile", category: "compliment-reply", tags: ["compliment reply", "funny", "ego"], tone: "funny" },
  { text: "you're one to talk, have you seen yourself lately?", category: "compliment-reply", tags: ["compliment reply", "return", "seen yourself"], tone: "flirty" },
  { text: "i didn't know i needed to hear that until just now", category: "compliment-reply", tags: ["compliment reply", "needed", "touched"], tone: "sincere" },

  // ═══════════════════════════════════════════════════════════════
  // 33. RUSSIAN LANGUAGE — OPENERS & CASUAL
  // ═══════════════════════════════════════════════════════════════

  { text: "привет, красотка, как дела?", category: "opener", tags: ["russian", "greeting", "casual", "beautiful"], tone: "sweet", language: "ru" },
  { text: "что делаешь? (а то я скучаю)", category: "conversation", tags: ["russian", "what doing", "miss", "casual"], tone: "sweet", language: "ru" },
  { text: "ну и как прошёл день, рассказывай", category: "opener", tags: ["russian", "day", "tell me", "casual"], tone: "sweet", language: "ru" },
  { text: "привет зайка, скучала по мне?", category: "opener", tags: ["russian", "bunny", "miss", "playful"], tone: "playful", language: "ru" },
  { text: "мне скучно, развлеки меня", category: "bored", tags: ["russian", "bored", "entertain", "playful"], tone: "playful", language: "ru" },
  { text: "чем занята? давай поболтаем", category: "opener", tags: ["russian", "busy", "chat", "casual"], tone: "sweet", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 34. RUSSIAN LANGUAGE — GOOD MORNING / GOOD NIGHT
  // ═══════════════════════════════════════════════════════════════

  { text: "доброе утро, красавица", category: "morning", tags: ["russian", "morning", "beautiful"], tone: "sweet", language: "ru" },
  { text: "доброе утро, солнышко мое", category: "morning", tags: ["russian", "morning", "sunshine", "pet name"], tone: "sweet", language: "ru" },
  { text: "с добрым утром, малыш. как спалось?", category: "morning", tags: ["russian", "morning", "how slept", "baby"], tone: "sweet", language: "ru" },
  { text: "утро доброе, ты первая о ком я подумал сегодня", category: "morning", tags: ["russian", "morning", "first thought"], tone: "romantic", language: "ru" },
  { text: "просыпайся, я уже по тебе скучаю", category: "morning", tags: ["russian", "morning", "wake up", "miss"], tone: "sweet", language: "ru" },
  { text: "спокойной ночи, мне будет сниться только ты", category: "night", tags: ["russian", "night", "dream"], tone: "romantic", language: "ru" },
  { text: "спокойной ночи, любимая. сладких снов", category: "night", tags: ["russian", "night", "sweet dreams", "loved one"], tone: "romantic", language: "ru" },
  { text: "сладких снов, малыш. завтра поговорим", category: "night", tags: ["russian", "night", "sweet dreams", "tomorrow"], tone: "sweet", language: "ru" },
  { text: "спи крепко, я рядом, хоть и далеко", category: "night", tags: ["russian", "night", "sleep", "close", "far"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 35. RUSSIAN LANGUAGE — COMPLIMENTS
  // ═══════════════════════════════════════════════════════════════

  { text: "ты сегодня особенно красивая", category: "compliment", tags: ["russian", "beautiful", "today", "compliment"], tone: "romantic", language: "ru" },
  { text: "ты мое солнышко ☀️", category: "compliment", tags: ["russian", "sunshine", "cute", "pet name"], tone: "sweet", language: "ru" },
  { text: "как можно быть такой милой?", category: "compliment", tags: ["russian", "cute", "question"], tone: "sweet", language: "ru" },
  { text: "ты такая, что слов нет — только эмоции", category: "compliment", tags: ["russian", "speechless", "emotions"], tone: "romantic", language: "ru" },
  { text: "у тебя самая красивая улыбка на свете", category: "compliment", tags: ["russian", "smile", "beautiful", "world"], tone: "romantic", language: "ru" },
  { text: "ты невероятная, и я каждый день в этом убеждаюсь", category: "compliment", tags: ["russian", "incredible", "every day"], tone: "sincere", language: "ru" },
  { text: "от твоих глаз невозможно оторваться", category: "compliment", tags: ["russian", "eyes", "impossible", "look away"], tone: "romantic", language: "ru" },
  { text: "мне нравится всё в тебе — от улыбки до характера", category: "compliment", tags: ["russian", "everything", "smile", "character"], tone: "sincere", language: "ru" },
  { text: "ты красивая не только снаружи, но и внутри", category: "compliment", tags: ["russian", "beautiful", "inside", "outside"], tone: "sincere", language: "ru" },
  { text: "с каждым днём ты всё красивее", category: "compliment", tags: ["russian", "more beautiful", "every day"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 36. RUSSIAN LANGUAGE — ROMANTIC / FEELINGS
  // ═══════════════════════════════════════════════════════════════

  { text: "ты заняла все мои мысли", category: "romantic", tags: ["russian", "thinking", "romantic", "all thoughts"], tone: "romantic", language: "ru" },
  { text: "хочу быть рядом с тобой прямо сейчас", category: "missing", tags: ["russian", "miss", "together", "right now"], tone: "romantic", language: "ru" },
  { text: "скучаю по тебе, приезжай скорее", category: "missing", tags: ["russian", "miss", "come", "soon"], tone: "sweet", language: "ru" },
  { text: "ты — лучшее, что со мной случилось", category: "romantic", tags: ["russian", "best thing", "happened"], tone: "romantic", language: "ru" },
  { text: "без тебя всё не то и все не те", category: "missing", tags: ["russian", "without you", "nothing same"], tone: "romantic", language: "ru" },
  { text: "я люблю тебя сильнее, чем вчера, но слабее, чем завтра", category: "romantic", tags: ["russian", "love", "more", "growing"], tone: "romantic", language: "ru" },
  { text: "мне с тобой так хорошо, что даже страшно", category: "romantic", tags: ["russian", "so good", "scared"], tone: "sincere", language: "ru" },
  { text: "ты делаешь меня лучше просто тем, что ты есть", category: "romantic", tags: ["russian", "make me better", "just by existing"], tone: "sincere", language: "ru" },
  { text: "рядом с тобой я забываю обо всём на свете", category: "romantic", tags: ["russian", "forget everything", "next to you"], tone: "romantic", language: "ru" },
  { text: "я выбираю тебя каждый день и это самое лёгкое решение в моей жизни", category: "romantic", tags: ["russian", "choose you", "every day", "easy decision"], tone: "sincere", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 37. RUSSIAN LANGUAGE — MISSING / LONG DISTANCE
  // ═══════════════════════════════════════════════════════════════

  { text: "скучаю по тебе невыносимо", category: "missing", tags: ["russian", "miss", "unbearable", "long distance"], tone: "romantic", language: "ru" },
  { text: "считаю дни до нашей встречи", category: "missing", tags: ["russian", "counting days", "meeting", "long distance"], tone: "romantic", language: "ru" },
  { text: "как же я хочу тебя обнять прямо сейчас", category: "missing", tags: ["russian", "hug", "right now", "miss"], tone: "romantic", language: "ru" },
  { text: "расстояние — это ничто, когда кто-то значит всё", category: "missing", tags: ["russian", "distance", "nothing", "means everything"], tone: "romantic", language: "ru" },
  { text: "ты далеко, но всегда в моём сердце", category: "missing", tags: ["russian", "far", "heart", "always"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 38. RUSSIAN LANGUAGE — FLIRTY / PLAYFUL
  // ═══════════════════════════════════════════════════════════════

  { text: "ты ведёшь себя опасно, и мне это нравится", category: "flirty", tags: ["russian", "dangerous", "like it", "flirty"], tone: "flirty", language: "ru" },
  { text: "прекрати быть такой красивой, я не могу сосредоточиться", category: "flirty", tags: ["russian", "beautiful", "can't focus", "flirty"], tone: "flirty", language: "ru" },
  { text: "ты в курсе, что ты сводишь меня с ума?", category: "flirty", tags: ["russian", "driving crazy", "question"], tone: "flirty", language: "ru" },
  { text: "если бы за каждую мысль о тебе мне давали рубль, я бы уже был миллионером", category: "flirty", tags: ["russian", "thinking", "ruble", "millionaire", "funny"], tone: "playful", language: "ru" },
  { text: "давай куда-нибудь сходим, только ты и я", category: "date", tags: ["russian", "date", "together", "just us"], tone: "sweet", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 39. RUSSIAN LANGUAGE — SUPPORT / BAD DAY
  // ═══════════════════════════════════════════════════════════════

  { text: "всё будет хорошо, я рядом", category: "support", tags: ["russian", "everything ok", "here", "support"], tone: "supportive", language: "ru" },
  { text: "расскажи мне, что случилось. я слушаю", category: "support", tags: ["russian", "tell me", "listening", "support"], tone: "supportive", language: "ru" },
  { text: "ты сильная, ты справишься. а я буду рядом", category: "support", tags: ["russian", "strong", "handle it", "here"], tone: "supportive", language: "ru" },
  { text: "плохой день — это временно. а мы с тобой — нет", category: "support", tags: ["russian", "bad day", "temporary", "we are not"], tone: "supportive", language: "ru" },
  { text: "хочешь — приеду, хочешь — позвоню, хочешь — просто помолчим вместе", category: "support", tags: ["russian", "come", "call", "silence together", "support"], tone: "supportive", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 40. RUSSIAN LANGUAGE — APOLOGY
  // ═══════════════════════════════════════════════════════════════

  { text: "прости меня, я был неправ. ты мне дороже любого спора", category: "apology", tags: ["russian", "sorry", "wrong", "more important"], tone: "sincere", language: "ru" },
  { text: "ненавижу, когда мы ссоримся. давай помиримся", category: "apology", tags: ["russian", "hate fighting", "make up"], tone: "sincere", language: "ru" },
  { text: "я облажался и я это знаю. дай мне шанс всё исправить", category: "apology", tags: ["russian", "messed up", "chance", "fix"], tone: "sincere", language: "ru" },
  { text: "мне очень жаль, что я тебя расстроил. это последнее, чего я хотел", category: "apology", tags: ["russian", "sorry", "upset", "last thing"], tone: "sincere", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 41. RUSSIAN LANGUAGE — JEALOUSY (playful)
  // ═══════════════════════════════════════════════════════════════

  { text: "это кто тебе лайк поставил? чисто из любопытства спрашиваю", category: "jealousy", tags: ["russian", "jealous", "like", "curious"], tone: "playful", language: "ru" },
  { text: "ты только моя, и точка", category: "jealousy", tags: ["russian", "jealous", "mine", "period"], tone: "playful", language: "ru" },
  { text: "ревную, но виду не подаю. ладно, подаю", category: "jealousy", tags: ["russian", "jealous", "pretend", "funny"], tone: "playful", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 42. RUSSIAN LANGUAGE — FUTURE / SERIOUS
  // ═══════════════════════════════════════════════════════════════

  { text: "я не представляю своё будущее без тебя", category: "future", tags: ["russian", "future", "can't imagine", "without you"], tone: "sincere", language: "ru" },
  { text: "куда бы жизнь ни привела — я хочу, чтобы ты была рядом", category: "future", tags: ["russian", "future", "wherever", "together"], tone: "romantic", language: "ru" },
  { text: "иногда думаю, какими мы будем через 5 лет. и улыбаюсь", category: "future", tags: ["russian", "future", "5 years", "smile"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 43. RUSSIAN LANGUAGE — EMOJI / CASUAL
  // ═══════════════════════════════════════════════════════════════

  { text: "улыбнись для меня 😊", category: "conversation", tags: ["russian", "smile", "cute", "emoji"], tone: "sweet", language: "ru" },
  { text: "ты моя 🫶", category: "emoji", tags: ["russian", "emoji", "mine", "casual"], tone: "sweet", language: "ru" },
  { text: "скучаюююю 😩❤️", category: "emoji", tags: ["russian", "emoji", "miss", "casual"], tone: "sweet", language: "ru" },
  { text: "иди сюдааа 🤗", category: "emoji", tags: ["russian", "emoji", "come here", "hug"], tone: "sweet", language: "ru" },
  { text: "ладно ты милая, признаю 😤❤️", category: "emoji", tags: ["russian", "emoji", "cute", "admit"], tone: "playful", language: "ru" },
  { text: "не могу перестать думать о тебе, это уже диагноз 😂💕", category: "emoji", tags: ["russian", "emoji", "thinking", "funny", "diagnosis"], tone: "playful", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 44. RUSSIAN LANGUAGE — PET NAMES
  // ═══════════════════════════════════════════════════════════════

  { text: "привет, зайка", category: "pet-name", tags: ["russian", "pet name", "bunny", "greeting"], tone: "sweet", language: "ru" },
  { text: "как дела, котёнок?", category: "pet-name", tags: ["russian", "pet name", "kitten", "how are you"], tone: "sweet", language: "ru" },
  { text: "малыш, ты где?", category: "pet-name", tags: ["russian", "pet name", "baby", "where"], tone: "sweet", language: "ru" },
  { text: "солнце моё, как ты?", category: "pet-name", tags: ["russian", "pet name", "sun", "how are you"], tone: "sweet", language: "ru" },
  { text: "любимая моя, думаю о тебе", category: "pet-name", tags: ["russian", "pet name", "loved one", "thinking"], tone: "romantic", language: "ru" },
  { text: "принцесса, скучаю", category: "pet-name", tags: ["russian", "pet name", "princess", "miss"], tone: "sweet", language: "ru" },
  { text: "ты мой ангел", category: "pet-name", tags: ["russian", "pet name", "angel"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 45. RUSSIAN LANGUAGE — HUMOR / MEMES
  // ═══════════════════════════════════════════════════════════════

  { text: "мой мозг в 3 часа ночи: а что если написать ей?", category: "humor", tags: ["russian", "meme", "3am", "brain", "text her"], tone: "funny", language: "ru" },
  { text: "я: надо сосредоточиться. тоже я: *думает о ней*", category: "humor", tags: ["russian", "meme", "focus", "thinking"], tone: "funny", language: "ru" },
  { text: "я бы завоевал для тебя мир, но пока могу только завтрак в постель", category: "humor", tags: ["russian", "conquer world", "breakfast", "funny"], tone: "playful", language: "ru" },
  { text: "уровень привязанности к тебе: проверяю телефон каждые 30 секунд", category: "humor", tags: ["russian", "meme", "attachment", "phone check"], tone: "funny", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 46. RUSSIAN LANGUAGE — DATE PLANNING
  // ═══════════════════════════════════════════════════════════════

  { text: "пошли куда-нибудь поужинать? я угощаю", category: "date", tags: ["russian", "dinner", "date", "treat"], tone: "sweet", language: "ru" },
  { text: "хочу приготовить для тебя ужин. что ты любишь?", category: "date", tags: ["russian", "cook", "dinner", "what you like"], tone: "sweet", language: "ru" },
  { text: "давай сбежим на выходные куда-нибудь подальше", category: "date", tags: ["russian", "escape", "weekend", "far away"], tone: "romantic", language: "ru" },
  { text: "кино, попкорн, ты рядом — идеальный вечер", category: "date", tags: ["russian", "movie", "popcorn", "perfect evening"], tone: "sweet", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 47. RUSSIAN LANGUAGE — LOVE REPLIES
  // ═══════════════════════════════════════════════════════════════

  { text: "конечно люблю, разве это не очевидно?", category: "love-reply", tags: ["russian", "love", "obvious", "of course"], tone: "playful", language: "ru" },
  { text: "люблю тебя больше, чем ты можешь себе представить", category: "love-reply", tags: ["russian", "love", "more than imagine"], tone: "sincere", language: "ru" },
  { text: "я люблю тебя. и это не просто слова", category: "love-reply", tags: ["russian", "love", "not just words"], tone: "sincere", language: "ru" },
  { text: "каждый день люблю тебя сильнее", category: "love-reply", tags: ["russian", "love", "every day", "stronger"], tone: "romantic", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 48. RUSSIAN LANGUAGE — PHOTO REPLIES
  // ═══════════════════════════════════════════════════════════════

  { text: "ты нереальная 😍", category: "photo-reply", tags: ["russian", "photo", "unreal", "emoji"], tone: "flirty", language: "ru" },
  { text: "как ты можешь быть такой красивой, это нечестно", category: "photo-reply", tags: ["russian", "photo", "beautiful", "unfair"], tone: "romantic", language: "ru" },
  { text: "это фото сделало мой день", category: "photo-reply", tags: ["russian", "photo", "made my day"], tone: "sweet", language: "ru" },
  { text: "подожди, я ещё раз посмотрю... да, всё ещё красавица", category: "photo-reply", tags: ["russian", "photo", "look again", "beautiful"], tone: "playful", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 49. RUSSIAN LANGUAGE — ANGRY / COLD SITUATIONS
  // ═══════════════════════════════════════════════════════════════

  { text: "не злись на меня, пожалуйста. давай поговорим", category: "angry", tags: ["russian", "mad", "please", "talk"], tone: "sincere", language: "ru" },
  { text: "я чувствую, что что-то не так. скажи мне, что происходит", category: "distant", tags: ["russian", "something wrong", "tell me"], tone: "sincere", language: "ru" },
  { text: "даже когда ты злишься, ты всё равно самая красивая", category: "angry", tags: ["russian", "mad", "still beautiful"], tone: "playful", language: "ru" },
  { text: "мне плохо, когда между нами всё не так. давай исправим это", category: "distant", tags: ["russian", "bad", "fix", "between us"], tone: "sincere", language: "ru" },

  // ═══════════════════════════════════════════════════════════════
  // 50. EXTRA — MIXED / OVERFLOW ENTRIES
  // ═══════════════════════════════════════════════════════════════

  { text: "you know what's crazy? i actually like you more every day, not less", category: "romantic", tags: ["crazy", "more", "every day"], tone: "sincere" },
  { text: "you're my person. that's it. that's the text", category: "romantic", tags: ["my person", "simple", "declaration"], tone: "sincere" },
  { text: "i didn't plan on falling for you this hard but here we are", category: "romantic", tags: ["falling", "didn't plan", "hard"], tone: "romantic" },
  { text: "you make me want to be better, and that's the realest compliment i can give", category: "romantic", tags: ["better", "real", "compliment"], tone: "sincere" },
  { text: "some people search their whole life for what i found in you", category: "romantic", tags: ["search", "whole life", "found"], tone: "romantic" },
  { text: "okay real talk — you're genuinely the kindest person i know", category: "compliment", tags: ["real talk", "kindest", "genuine"], tone: "sincere" },
  { text: "i would cross oceans for you. or at the very least, this city. in traffic", category: "funny", tags: ["oceans", "traffic", "funny", "effort"], tone: "funny" },
  { text: "you are home to me", category: "romantic", tags: ["home", "simple", "deep"], tone: "sincere" },
  { text: "if i had to choose between breathing and loving you, i'd use my last breath to say i love you", category: "cheesy", tags: ["cheesy", "breathing", "love", "last breath"], tone: "cheesy" },
  { text: "the fact that you exist in the same timeline as me feels like a miracle", category: "romantic", tags: ["exist", "timeline", "miracle"], tone: "romantic" },
  { text: "you're not just a chapter in my life, you're the whole story", category: "romantic", tags: ["chapter", "story", "whole"], tone: "romantic" },
  { text: "i've never been good with words but with you i want to try", category: "romantic", tags: ["words", "try", "honest"], tone: "sincere" },
  { text: "every time i think i can't love you more, you prove me wrong", category: "romantic", tags: ["love", "more", "prove wrong"], tone: "romantic" },
  { text: "you make my heart do that stupid thing where it beats faster", category: "flirty", tags: ["heart", "beats faster", "cute"], tone: "sweet" },
  { text: "honestly? you're the first person i want to share good news with", category: "romantic", tags: ["first person", "good news", "share"], tone: "sincere" },
  { text: "i'm not perfect but i'm perfectly into you", category: "flirty", tags: ["not perfect", "into you", "pun"], tone: "flirty" },
  { text: "our conversations are my favorite thing about my day", category: "romantic", tags: ["conversations", "favorite", "day"], tone: "sweet" },
  { text: "the way you say my name hits different", category: "flirty", tags: ["name", "hits different", "voice"], tone: "flirty" },
  { text: "have i told you today that you're amazing? no? well, you're amazing", category: "compliment", tags: ["amazing", "today", "reminder"], tone: "sweet" },
  { text: "ты для меня всё", category: "romantic", tags: ["russian", "you are everything", "simple"], tone: "sincere", language: "ru" },
  { text: "с тобой даже молчание уютное", category: "romantic", tags: ["russian", "silence", "comfortable", "together"], tone: "sincere", language: "ru" },
  { text: "ты — причина моей улыбки", category: "romantic", tags: ["russian", "reason", "smile"], tone: "sweet", language: "ru" },
  { text: "обещаю всегда быть честным с тобой", category: "romantic", tags: ["russian", "promise", "honest", "always"], tone: "sincere", language: "ru" },
  { text: "знаешь что? я самый счастливый, потому что у меня есть ты", category: "romantic", tags: ["russian", "happiest", "because of you"], tone: "sincere", language: "ru" },
];
